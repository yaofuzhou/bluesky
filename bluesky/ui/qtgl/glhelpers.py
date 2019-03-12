"""
BlueSky-QtGL tools       : Tools and objects that are used in the BlueSky-QtGL implementation

Methods:
    load_texture(fname)  : GL-texture load function. Returns id of new texture
    ShaderProgram()     : Constructor of a BlueSky shader program object: the main shader object in BlueSky-QtGL


Internal methods and classes:
    create_font_array()
    create_attrib_from_glbuf()
    create_attrib_from_nparray()


Created by    : Joost Ellerbroek
Date          : April 2015

Modification  :
By            :
Date          :
------------------------------------------------------------------
"""
import os
import ctypes
from collections import namedtuple
try:
    from collections.abc import Collection, MutableMapping
except ImportError:
    # In python <3.3 collections.abc doesn't exist
    from collections import Collection, MutableMapping

from PyQt5.QtGui import QImage, QOpenGLTexture, QOpenGLBuffer
import OpenGL.GL as gl
import numpy as np
from bluesky import settings

from bluesky.ui.qtgl.dds import DDSTexture
msg1282 = False # GL error 1282 when quitting should only be reported once

# Register settings defaults
settings.set_variable_defaults(gfx_path='data/graphics')


class Texture(QOpenGLTexture):
    def __init__(self, fname=None):
        super().__init__(QOpenGLTexture.Target2D)
        if fname:
            self.load(fname)

    def load(self, fname):
        if fname[-3:].lower() == 'dds':
            tex = DDSTexture(fname)
            self.setFormat(QOpenGLTexture.RGB_DXT1)
            self.setSize(tex.width, tex.height)
            self.setWrapMode(QOpenGLTexture.Repeat)
            self.allocateStorage()
            self.setCompressedData(len(tex.data), tex.data)
        else:
            self.setData(QImage(fname))
            self.setWrapMode(QOpenGLTexture.Repeat)


class GLBuffer:
    ubo_max_binding = 1

    ''' GL buffer convenience wrapper. '''
    def __init__(self, size=None, target=gl.GL_ARRAY_BUFFER, usage=gl.GL_STATIC_DRAW, data=None):
        if size is None and data is None:
            raise ValueError('Either a size or a set of data should be provided when creating a GL buffer')
        self.target = target
        self.usage = usage
        dbuf, dsize = GLBuffer._raw(data)
        self.buf_size = size or dsize
        self.buf_id = gl.glGenBuffers(1)
        gl.glBindBuffer(self.target, self.buf_id)
        gl.glBufferData(self.target, self.buf_size, dbuf, self.usage)

    def bind(self):
        ''' Bind this buffer to the current context. '''
        gl.glBindBuffer(self.target, self.buf_id)

    def update(self, data, offset=0, size=None):
        ''' Send new data to this GL buffer. '''
        dbuf, dsize = GLBuffer._raw(data)
        size = size or dsize
        if size > self.buf_size:
            print('GLBuffer: Warning, trying to send more data to buffer than allocated size.')
        gl.glBindBuffer(self.target, self.buf_id)
        gl.glBufferSubData(self.target, offset, min(self.buf_size, size), dbuf)
        # TODO: master branch has try/except for buffer writes after closing context

    @staticmethod
    def _raw(data):
        if isinstance(data, np.ndarray):
            return data, data.nbytes
        if isinstance(data, (ctypes.Structure, ctypes.Array)):
            return ctypes.pointer(data), ctypes.sizeof(data)
        return None, 0

    @classmethod
    def createubo(cls, size):
        ubo = cls(size, gl.GL_UNIFORM_BUFFER, gl.GL_STREAM_DRAW)
        ubo.binding = GLBuffer.ubo_max_binding
        GLBuffer.ubo_max_binding += 1
        gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, ubo.binding, ubo.buf_id)
        return ubo


class ShaderSet(MutableMapping):
    ''' A set of shader programs for BlueSky.

        Convenience class to easily switch between sets of shader programs
        (e.g., between the radarwidget and the nd.)

        Normally, each set contains at least the following programs:
        'normal':   Rendering of regular, untextured shapes
        'textured': Rendering of regular, textured shapes
        'text':     Rendering of text objects
        'ssd':      Rendering of SSD objects
    '''
    # Currently selected shader set
    selected = None

    def __init__(self):
        super().__init__()
        self._programs = dict()
        self._ubos = dict()
        self._spath = ''
        if ShaderSet.selected is None:
            self.select()

    def select(self):
        ''' Select this shader set. '''
        ShaderSet.selected = self

    def update_ubo(self, uboname, *args, **kwargs):
        ''' Update an uniform buffer object of this shader set. '''
        ubo = self._ubos.get(uboname, None)
        if not ubo:
            raise KeyError('Uniform Buffer Object', uboname, 'not found in shader set.')
        ubo.update(*args, **kwargs)

    def set_shader_path(self, path):
        ''' Set a search path for shader files. '''
        self._spath = path

    def load_shader(self, shader_name, vs, fs, gs=None, *args, **kwargs):
        ''' Load a shader into this shader set.
            default shader names are: normal, textured, and text. '''
        vs = os.path.join(self._spath, vs)
        fs = os.path.join(self._spath, fs)
        if gs:
            gs = os.path.join(self._spath, gs)
        self[shader_name] = ShaderProgram(vs, fs, gs, *args, **kwargs)

    def __getitem__(self, key):
        ret = self._programs.get(key, None)
        if not ret:
            raise KeyError('Shader program', key, 'not found in shader set.')
        return ret

    def __setitem__(self, key, program):
        if not isinstance(program, ShaderProgram):
            raise ValueError('Only ShaderProgram objects can be added to a ShaderSet')
        self._programs[key] = program
        # Bind UBO buffers of this shader set to program's UBO's
        for name, size in program.ubos.items():
            ubo = self._ubos.get(name, None)
            if ubo is None:
                ubo = GLBuffer.createubo(size)
                self._ubos[name] = ubo

            program.bind_uniform_buffer(name, ubo)

    def __delitem__(self, key):
        del(self._programs[key])

    def __iter__(self):
        return iter(self._programs)

    def __len__(self):
        return len(self._programs)


class ShaderProgram():
    used_program = None

    def __init__(self, vertex_shader, fragment_shader, geom_shader=None):
        self.shaders = list()
        self.attribs = dict()
        self.ubos    = dict()
        GLVariable = namedtuple('GLVariable', ['loc', 'size'])

        # Compile shaders and link program
        self.compile_shader(vertex_shader, gl.GL_VERTEX_SHADER)
        self.compile_shader(fragment_shader, gl.GL_FRAGMENT_SHADER)
        if geom_shader:
            self.compile_shader(geom_shader, gl.GL_GEOMETRY_SHADER)
        self.link()

        size = gl.GLint()
        name = (ctypes.c_char * 20)()

        # Obtain list of attributes with location and size info
        n_attrs = gl.glGetProgramiv(self.program, gl.GL_ACTIVE_ATTRIBUTES)
        for a in range(n_attrs):
            atype = gl.GLint()
            gl.glGetActiveAttrib(self.program, a, 20, None,
                ctypes.pointer(size), ctypes.pointer(atype), ctypes.pointer(name))
            loc = gl.glGetAttribLocation(self.program, name.value)
            typesize = _glvar_sizes.get(atype.value, 1)
            self.attribs[name.value.decode('utf8')] = GLVariable(loc, size.value * typesize)

        n_uniforms = gl.glGetProgramiv(self.program, gl.GL_ACTIVE_UNIFORMS)
        all_uids = set(range(n_uniforms))

        n_ub = gl.glGetProgramiv(self.program, gl.GL_ACTIVE_UNIFORM_BLOCKS)
        for ub in range(n_ub):
            gl.glGetActiveUniformBlockName(self.program, ub, 20, None, ctypes.pointer(name))
            gl.glGetActiveUniformBlockiv(self.program, ub, gl.GL_UNIFORM_BLOCK_DATA_SIZE, ctypes.pointer(size))
            self.ubos[name.value.decode('utf-8')] = size.value
            ubsize = gl.GLint()
            gl.glGetActiveUniformBlockiv(self.program, ub, gl.GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS, ctypes.pointer(ubsize))
            indices = (ctypes.c_int * ubsize.value)()
            gl.glGetActiveUniformBlockiv(self.program, ub, gl.GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, ctypes.pointer(indices))
            all_uids -= set(indices)
            # print('Uniform block: ', name.value.decode('utf8'), 'size =', ubsize.value)
            # for i in indices:
            #     usize = gl.GLint()
            #     utype = gl.GLint()
            #     uname = (ctpes.c_char * 20)()
            #     # gl.glGetActiveUniform(self.program, i, 20, None, ctypes.pointer(usize), ctypes.pointer(utype), ctypes.pointer(uname))
            #     # print('block uniform', i, '=', uname.value.decode('utf8'), 'size =', usize.value)
            #     print(gl.glGetActiveUniform(self.program, i))

            for u in all_uids:
                name, size, utype = gl.glGetActiveUniform(self.program, u)
                setattr(self, name.decode('utf-8'), GLVariable(u, size * _glvar_sizes.get(utype, 1)))

    def compile_shader(self, fname, type):
        """Compile a vertex shader from source."""
        source = open(fname, 'r').read()
        shader = gl.glCreateShader(type)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        # check compilation error
        result = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
        if not(result):
            raise RuntimeError(gl.glGetShaderInfoLog(shader))
            gl.glDeleteShader(shader)
            return

        self.shaders.append(shader)

    def link(self):
        """Create a shader program with from compiled shaders."""
        self.program = gl.glCreateProgram()
        for i in range(0, len(self.shaders)):
            gl.glAttachShader(self.program, self.shaders[i])

        gl.glLinkProgram(self.program)
        # check linking error
        result = gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS)
        if not(result):
            raise RuntimeError(gl.glGetProgramInfoLog(self.program))
            gl.glDeleteProgram(self.program)
            for i in range(0, len(self.shaders)):
                gl.glDeleteShader(self.shaders[i])

        # Clean up
        for i in range(0, len(self.shaders)):
            gl.glDetachShader(self.program, self.shaders[i])
            gl.glDeleteShader(self.shaders[i])

    def use(self):
        if ShaderProgram.used_program != self:
            gl.glUseProgram(self.program)
            ShaderProgram.used_program = self

    def bind_uniform_buffer(self, ubo_name, ubo):
        idx = gl.glGetUniformBlockIndex(self.program, ubo_name)
        gl.glUniformBlockBinding(self.program, idx, ubo.binding)


class VertexAttributeObject(object):
    bound_vao = -1

    class Attrib:
        def __init__(self, name, loc, size, parent):
            self.name = name
            self.loc = loc
            self.size = size
            self.parent = parent
            self.buf = None
            self.divisor = 0
            self.datatype = gl.GL_FLOAT
            self.stride = 0
            self.offset = None
            self.normalize = False

        def enable(self):
            gl.glEnableVertexAttribArray(self.loc)

        def update(self, *args, **kwargs):
            self.buf.update(*args, **kwargs)

        def bind(self, data, usage=gl.GL_STATIC_DRAW, instance_divisor=0, datatype=gl.GL_FLOAT, stride=0, offset=None, normalize=False):
            if VertexAttributeObject.bound_vao is not self.parent.vao_id:
                gl.glBindVertexArray(self.parent.vao_id)
                VertexAttributeObject.bound_vao = self.parent.vao_id

            self.divisor = instance_divisor
            self.datatype = gl.GL_UNSIGNED_BYTE if self.name == 'color' else datatype
            self.stride = stride
            self.offset = offset
            self.normalize = True  if self.name == 'color' else normalize

            # Keep track of max instance divisor
            self.parent.max_instance_divisor = max(instance_divisor, self.parent.max_instance_divisor)

            # If the input is an array create a new GL buffer, otherwise assume the buffer already exists and a buffer ID is passed
            if isinstance(data, Collection):
                # Color attribute has a special condition for a single color
                if self.name =='color' and np.size(data) in (3, 4):
                    # Add full alpha if none is given
                    self.parent.single_color = np.append(data, 255) if len(data) == 3 else data
                    return
                # Get an index to one new buffer in GPU mem, bind it, and copy the array data to it
                self.buf = GLBuffer(data=data, usage=usage)
            elif isinstance(data, int):
                self.buf = GLBuffer(size=data, usage=usage)
            elif isinstance(data, GLBuffer):
                # A GL buffer is passed
                self.buf = data
                self.buf.bind()
            elif isinstance(data, VertexAttributeObject.Attrib):
                # Should bind to same buffer as passed attrib
                self.buf = data.buf
                self.buf.bind()
            else:
                raise ValueError('Unknown datatype passed.')

            # Assign this buffer to one of the attributes in the shader
            gl.glEnableVertexAttribArray(self.loc)
            gl.glVertexAttribPointer(self.loc, self.size, self.datatype, self.normalize, self.stride, self.offset)
            # For instanced data, indicate per how many instances we move a step in the buffer (1=per instance)
            if instance_divisor > 0:
                gl.glVertexAttribDivisor(self.loc, self.divisor)
            # Clean up
            gl.glDisableVertexAttribArray(self.loc)
            # Add this attribute to enabled attributes
            self.parent.enabled_attributes.append(self)

    def __init__(self, primitive_type=None, n_instances=0, shader_type='normal', texture=None, **attribs):
        # Get attributes for the target shader type
        self.shader_type = shader_type
        for name, attr in ShaderSet.selected[self.shader_type].attribs.items():
            setattr(self, name, VertexAttributeObject.Attrib(name, attr.loc, attr.size, self))
        
        self.vao_id = gl.glGenVertexArrays(1)
        self.enabled_attributes = list()
        self.primitive_type = primitive_type
        self.first_vertex = 0
        self.vertex_count = 0
        self.n_instances = n_instances
        self.max_instance_divisor = 0
        self.single_color = None

        # Set texture if passed
        self.texture = None
        if texture:
            self.texture = Texture(texture)
            if shader_type == 'normal':
                self.shader_type = 'textured'

        # Set passed attributes
        self.set_attribs(**attribs)

    def set_primitive_type(self, primitive_type):
        self.primitive_type = primitive_type

    def set_vertex_count(self, count):
        self.vertex_count = int(count)

    def set_first_vertex(self, vertex):
        self.first_vertex = vertex

    def update(self, **attribs):
        ''' Update one or more buffers for this object. '''
        for name, data in attribs.items():
            attrib = getattr(self, name, None)
            if not isinstance(attrib, VertexAttributeObject.Attrib):
                raise KeyError('Unknown attribute ' + name)
            # Special attribs: color and vertex
            if name == 'vertex' and isinstance(data, Collection):
                self.vertex_count = np.size(data) // 2

            # Update the buffer of the attribute
            attrib.update(data)

    def set_attribs(self, usage=gl.GL_STATIC_DRAW, instance_divisor=0, datatype=gl.GL_FLOAT, stride=0, offset=None, normalize=False, **attribs):
        for name, data in attribs.items():
            attrib = getattr(self, name, None)
            if not isinstance(attrib, VertexAttributeObject.Attrib):
                raise KeyError('Unknown attribute ' + name + ' for shader type ' + self.shader_type)
            # Special attribs: color and vertex
            if name == 'vertex' and isinstance(data, Collection):
                self.vertex_count = np.size(data) // 2
            attrib.bind(data, usage, instance_divisor, datatype, stride, offset, normalize)

    def bind(self):
        if VertexAttributeObject.bound_vao != self.vao_id:
            gl.glBindVertexArray(self.vao_id)
            VertexAttributeObject.bound_vao = self.vao_id
            for attrib in self.enabled_attributes:
                attrib.enable()

    def draw(self, primitive_type=None, first_vertex=None, vertex_count=None, n_instances=None):
        if primitive_type is None:
            primitive_type = self.primitive_type

        if first_vertex is None:
            first_vertex = self.first_vertex

        if vertex_count is None:
            vertex_count = self.vertex_count

        if n_instances is None:
            n_instances = self.n_instances

        if vertex_count == 0:
            return

        ShaderSet.selected[self.shader_type].use()
        self.bind()

        if self.single_color is not None:
            gl.glVertexAttrib4Nub(self.color.loc, *self.single_color)
        elif self.texture:
            self.texture.bind()

        if n_instances > 0:
            gl.glDrawArraysInstanced(primitive_type, first_vertex, vertex_count, n_instances * self.max_instance_divisor)
        else:
            gl.glDrawArrays(primitive_type, first_vertex, vertex_count)

    @staticmethod
    def unbind_all():
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        VertexAttributeObject.bound_vao = -1

    @classmethod
    def copy(cls, original):
        """ Copy a render object from one context to the other.
        """
        new = VertexAttributeObject(original.primitive_type, original.n_instances, original.shader_type)
        new.first_vertex = original.first_vertex
        new.vertex_count = original.vertex_count

        # Bind the same attributes for the new renderobject
        # [size, buf_id, instance_divisor, datatype]
        for attrib in original.enabled_attributes:
            getattr(new, attrib.name).bind(attrib,
                instance_divisor=attrib.divisor, datatype=attrib.datatype,
                stride=attrib.stride, offset=attrib.offset,
                normalize=attrib.normalize)

        # Copy possible object attributes that were added to the renderobject
        for attr, val in original.__dict__.items():
            if attr not in new.__dict__:
                setattr(new, attr, val)

        return new


class RenderObject:
    ''' Convenience class for drawing different (nested) objects. '''

    def __init__(self):
        self.children = list()

    def draw(self, *args, **kwargs):
        for child in self.children:
            child.draw(*args, **kwargs)


class Circle(VertexAttributeObject):
    ''' Convenience class for a circle. '''

    def __init__(self, radius, nsegments=36, *args, **kwargs):
        vcircle = np.transpose(np.array((
            radius * np.cos(np.linspace(0.0, 2.0 * np.pi, nsegments)),
            radius * np.sin(np.linspace(0.0, 2.0 * np.pi, nsegments))),
            dtype=np.float32))
        super().__init__(gl.GL_LINE_LOOP, vertex=vcircle, *args, **kwargs)


class Rectangle(VertexAttributeObject):
    ''' Convenience class for a rectangle. '''
    def __init__(self, w, h, fill=False, *args, **kwargs):
        primitive_type = gl.GL_TRIANGLE_FAN if fill else gl.GL_LINE_LOOP
        vrect = np.array([(-0.5 * h, 0.5 * w), (-0.5 * h, -0.5 * w),
                          (0.5 * h, -0.5 * w), (0.5 * h, 0.5 * w)], dtype=np.float32)
        super().__init__(primitive_type, vertex=vrect, *args, **kwargs)


class Font(object):
    def __init__(self, tex_id=0, char_ar=1.0):
        self.tex_id         = tex_id
        self.loc_char_size  = 0
        self.loc_block_size = 0
        self.char_ar        = char_ar

    def copy(self):
        return Font(self.tex_id, self.char_ar)

    def init_shader(self, program):
        self.loc_char_size = gl.glGetUniformLocation(program.program, 'char_size')
        self.loc_block_size = gl.glGetUniformLocation(program.program, 'block_size')

    def use(self):
        gl.glActiveTexture(gl.GL_TEXTURE0 + 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.tex_id)

    def set_char_size(self, char_size):
        gl.glUniform2f(self.loc_char_size, char_size, char_size * self.char_ar)

    def set_block_size(self, block_size):
        gl.glUniform2i(self.loc_block_size, block_size[0], block_size[1])

    def create_font_array(self):
        # Load the first image to get font size
        img          = QImage(os.path.join(settings.gfx_path, 'font/32.png'))
        imgsize      = (img.width(), img.height())
        self.char_ar = float(imgsize[1]) / imgsize[0]

        # Set-up the texture array
        self.tex_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.tex_id)
        gl.glTexImage3D(gl.GL_TEXTURE_2D_ARRAY, 0, gl.GL_RGBA8, imgsize[0], imgsize[1], 127 - 30, 0, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameterf(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
        gl.glTexParameterf(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
        # We're using the ASCII range 32-126; space, uppercase, lower case, numbers, brackets, punctuation marks
        for i in range(30, 127):
            img = QImage(os.path.join(settings.gfx_path, 'font/%d.png' % i)).convertToFormat(QImage.Format_ARGB32)
            ptr = ctypes.c_void_p(int(img.constBits()))
            gl.glTexSubImage3D(gl.GL_TEXTURE_2D_ARRAY, 0, 0, 0, i - 30, imgsize[0], imgsize[1], 1, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE, ptr)

    @staticmethod
    def char(x, y, w, h, c=32):
        # Two triangles per character
        vertices  = [(x, y + h), (x, y), (x + w, y + h), (x + w, y + h), (x, y), (x + w, y)]
        texcoords = [(0, 0, c), (0, 1, c), (1, 0, c), (1, 0, c), (0, 1, c), (1, 1, c)]
        return vertices, texcoords

    def prepare_text_string(self, text_string, char_size=16.0, text_color=(0.0, 1.0, 0.0), vertex_offset=(0.0, 0.0)):
        ret = VertexAttributeObject(gl.GL_TRIANGLES, shader_type='text')

        vertices, texcoords = [], []
        w, h = char_size, char_size * self.char_ar
        x, y = vertex_offset
        for i, c in enumerate(text_string):
            v, t = self.char(x + i * w, y, w, h, ord(c))
            vertices  += v
            texcoords += t

        ret.set_attribs(vertex=np.array(vertices, dtype=np.float32),
                        texcoords=np.array(texcoords, dtype=np.float32),
                        color=np.array(text_color, dtype=np.uint8))

        ret.char_size  = char_size
        ret.block_size = (len(text_string), 1)
        return ret

    def prepare_text_instanced(self, text_array, textblock_size, origin_lat=None, origin_lon=None, text_color=None, char_size=16.0, vertex_offset=(0.0, 0.0)):
        ret       = VertexAttributeObject(gl.GL_TRIANGLES, shader_type='text')
        w, h      = char_size, char_size * self.char_ar
        x, y      = vertex_offset
        v, t      = self.char(x, y, w, h)
        vertices  = v
        texcoords = t
        ret.set_attribs(vertex=np.array(vertices, dtype=np.float32),
                        texcoords=np.array(texcoords, dtype=np.float32))

        ret.texdepth.bind(text_array, instance_divisor=1, datatype=gl.GL_UNSIGNED_BYTE)
        divisor = textblock_size[0] * textblock_size[1]
        if origin_lat is not None:
            ret.lat.bind(origin_lat, instance_divisor=divisor)
        if origin_lon is not None:
            ret.lon.bind(origin_lon, instance_divisor=divisor)

        if text_color is not None:
            ret.color.bind(text_color, instance_divisor=divisor)

        ret.block_size = textblock_size
        ret.char_size = char_size

        return ret

_glvar_sizes = {
    gl.GL_FLOAT: 1, gl.GL_FLOAT_VEC2: 2, gl.GL_FLOAT_VEC3: 3,
    gl.GL_FLOAT_VEC4: 4, gl.GL_FLOAT_MAT2: 4, gl.GL_FLOAT_MAT3: 9,
    gl.GL_FLOAT_MAT4: 16, gl.GL_FLOAT_MAT2x3: 6, gl.GL_FLOAT_MAT2x4: 8,
    gl.GL_FLOAT_MAT3x2: 6, gl.GL_FLOAT_MAT3x4: 12, gl.GL_FLOAT_MAT4x2: 8,
    gl.GL_FLOAT_MAT4x3: 12, gl.GL_INT: 1, gl.GL_INT_VEC2: 2, gl.GL_INT_VEC3: 3,
    gl.GL_INT_VEC4: 4, gl.GL_UNSIGNED_INT: 1, gl.GL_UNSIGNED_INT_VEC2: 2,
    gl.GL_UNSIGNED_INT_VEC3: 3, gl.GL_UNSIGNED_INT_VEC4: 4, gl.GL_DOUBLE: 1,
    gl.GL_DOUBLE_VEC2: 2, gl.GL_DOUBLE_VEC3: 3, gl.GL_DOUBLE_VEC4: 4,
    gl.GL_DOUBLE_MAT2: 4, gl.GL_DOUBLE_MAT3: 9, gl.GL_DOUBLE_MAT4: 16,
    gl.GL_DOUBLE_MAT2x3: 6, gl.GL_DOUBLE_MAT2x4: 8, gl.GL_DOUBLE_MAT3x2: 6,
    gl.GL_DOUBLE_MAT3x4: 12, gl.GL_DOUBLE_MAT4x2: 8, gl.GL_DOUBLE_MAT4x3: 12}
