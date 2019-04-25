from os import path
from PyQt5.QtCore import Qt, QEvent, qCritical, QTimer, QT_VERSION
from PyQt5.QtOpenGL import QGLWidget
from ctypes import c_float, c_int, Structure
import numpy as np
import OpenGL.GL as gl

import bluesky as bs
from bluesky import settings
from bluesky.ui import palette
from bluesky.ui.radarclick import radarclick
from bluesky.ui.qtgl import console
from bluesky.ui.qtgl.customevents import ACDataEvent, RouteDataEvent
from bluesky.tools.aero import ft, nm, kts
from bluesky.tools import geo
from bluesky.navdatabase import load_aptsurface, load_coastlines
from .glhelpers import ShaderSet, ShaderProgram, VertexAttributeObject, Font, \
    GLBuffer, Circle, Rectangle, Texture


# Register settings defaults
settings.set_variable_defaults(
    gfx_path='data/graphics',
    text_size=13, apt_size=10,
    wpt_size=10, ac_size=16,
    asas_vmin=200.0, asas_vmax=500.0)

palette.set_default_colors(
    aircraft=(0,255,0),
    aptlabel=(220, 250, 255),
    aptsymbol=(148, 178, 235),
    background=(0,0,0),
    coastlines=(85, 85, 115),
    conflict=(255, 160, 0),
    pavement=(160, 160, 160),
    polys=(0,0,255),
    previewpoly=(0, 204, 255),
    route=(255, 0, 255),
    runways=(100, 100, 100),
    taxiways=(100, 100, 100),
    thresholds=(255,255,255),
    trails=(0, 255, 255),
    wptlabel=(220, 250, 255),
    wptsymbol=(148, 178, 235)
)

# Static defines
MAX_NAIRCRAFT = 10000
MAX_NCONFLICTS = 25000
ROUTE_SIZE = 500
POLYPREV_SIZE = 100
POLY_SIZE = 2000
CUSTWP_SIZE = 1000
TRAILS_SIZE = MAX_NAIRCRAFT * 1000

REARTH_INV = 1.56961231e-7

VERTEX_IS_LATLON, VERTEX_IS_METERS, VERTEX_IS_SCREEN = list(range(3))

# Qt smaller than 5.6.2 needs a different approach to pinch gestures
CORRECT_PINCH = False
if QT_VERSION <= 0x050600:
    import platform
    CORRECT_PINCH = platform.system() == 'Darwin'


class radarShaders(ShaderSet):
    def __init__(self):
        super().__init__()
        class GlobalData(Structure):
            _fields_ = [("wrapdir", c_int), ("wraplon", c_float), ("panlat", c_float), ("panlon", c_float),
            ("zoom", c_float), ("screen_width", c_int), ("screen_height", c_int), ("vertex_scale_type", c_int)]
        self.data = GlobalData()

    def load_shaders(self):
        self.set_shader_path(path.join(settings.gfx_path, 'shaders'))
        # Load all shaders for this shader set
        self.load_shader('normal', 'radarwidget-normal.vert', 'radarwidget-color.frag')
        self.load_shader('textured', 'radarwidget-normal.vert', 'radarwidget-texture.frag')
        self.load_shader('text', 'radarwidget-text.vert', 'radarwidget-text.frag')
        self.load_shader('ssd', 'ssd.vert', 'ssd.frag', 'ssd.geom')

    def set_wrap(self, wraplon, wrapdir):
        self.data.wrapdir = wrapdir
        self.data.wraplon = wraplon

    def set_pan_and_zoom(self, panlat, panlon, zoom):
        self.data.panlat = panlat
        self.data.panlon = panlon
        self.data.zoom = zoom

    def set_win_width_height(self, w, h):
        self.data.screen_width = w
        self.data.screen_height = h

    def enable_wrap(self, flag=True):
        if not flag:
            wrapdir = self.data.wrapdir
            self.data.wrapdir = 0
            self.update_ubo('global_data', self.data, 0, 4)
            self.data.wrapdir = wrapdir
        else:
            self.update_ubo('global_data', self.data, 0, 4)

    def set_vertex_scale_type(self, vertex_scale_type):
        self.data.vertex_scale_type = vertex_scale_type
        self.update_ubo('global_data', self.data)

class RadarWidget(QGLWidget):
    def __init__(self, shareWidget=None):
        self.shaderset = radarShaders()
        self.width = self.height = 600
        self.viewport = (0, 0, 600, 600)
        self.panlat = 0.0
        self.panlon = 0.0
        self.zoom = 1.0
        self.ar = 1.0
        self.flat_earth = 1.0
        self.wraplon = int(-999)
        self.wrapdir = int(0)

        self.map_texture = Texture()
        self.naircraft = 0
        self.nwaypoints = 0
        self.ncustwpts = 0
        self.nairports = 0
        self.route_acid = ""
        self.apt_inrange = np.array([])
        self.asas_vmin = settings.asas_vmin
        self.asas_vmax = settings.asas_vmax
        self.initialized = False

        self.acdata = ACDataEvent()
        self.routedata = RouteDataEvent()

        self.panzoomchanged = False
        self.mousedragged = False
        self.mousepos = (0, 0)
        self.prevmousepos = (0, 0)

        # Load vertex data
        self.vbuf_asphalt, self.vbuf_concrete, self.vbuf_runways, self.vbuf_rwythr, \
            self.apt_ctrlat, self.apt_ctrlon, self.apt_indices = load_aptsurface()
        self.coastvertices, self.coastindices = load_coastlines()

        # Only initialize super class after loading data to avoid Qt starting
        # things before we are ready.
        super(RadarWidget, self).__init__(shareWidget=shareWidget)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.grabGesture(Qt.PanGesture)
        self.grabGesture(Qt.PinchGesture)
        # self.grabGesture(Qt.SwipeGesture)
        self.setMouseTracking(True)

        # Connect to the io client's activenode changed signal
        console.cmdline_stacked.connect(self.cmdline_stacked)
        bs.net.actnodedata_changed.connect(self.actnodedataChanged)
        bs.net.stream_received.connect(self.on_simstream_received)

    def actnodedataChanged(self, nodeid, nodedata, changed_elems):
        ''' Update buffers when a different node is selected, or when
            the data of the current node is updated. '''
        self.makeCurrent()

        # Shape data change
        if 'SHAPE' in changed_elems:
            if nodedata.polys:
                contours, fills, colors = zip(*nodedata.polys.values())
                # Create contour buffer with color
                self.allpolys.update(vertex=np.concatenate(contours),
                                     color=np.concatenate(colors))

                # Create fill buffer
                self.allpfill.update(vertex=np.concatenate(fills))
            else:
                self.allpolys.set_vertex_count(0)
                self.allpfill.set_vertex_count(0)

        # Trail data change
        if 'TRAILS' in changed_elems:
            if len(nodedata.traillat0):
                self.traillines.update(vertex=np.array(
                    list(zip(nodedata.traillat0, nodedata.traillon0,
                             nodedata.traillat1, nodedata.traillon1)), dtype=np.float32))

        if 'CUSTWPT' in changed_elems:
            if nodedata.custwplbl:
                self.customwp.update(lat=nodedata.custwplat,
                                     lon=nodedata.custwplon)
                self.custwplblbuf.update(np.array(nodedata.custwplbl, dtype=np.string_))
            self.ncustwpts = len(nodedata.custwplat)

        # Update pan/zoom
        if 'PANZOOM' in changed_elems:
            self.panzoom(pan=nodedata.pan, zoom=nodedata.zoom, absolute=True)

    def create_objects(self):
        text_size = settings.text_size
        apt_size = settings.apt_size
        wpt_size = settings.wpt_size
        ac_size = settings.ac_size

        # Initialize font for radar view with specified settings
        self.font = Font()
        self.font.create_font_array()
        self.font.init_shader(self.text_shader)

        # Load and bind world texture
        max_texture_size = gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE)
        print('Maximum supported texture size: %d' % max_texture_size)
        map_texname = ''
        for i in [16384, 8192, 4096]:
            if max_texture_size >= i:
                fname = path.join(settings.gfx_path, 'world.%dx%d.dds' % (i, i // 2))
                print('Loading texture ' + fname)
                # self.map_texture.load(fname)
                map_texname = fname
                break

        # Create initial empty buffers for aircraft position, orientation, label, and color
        # usage flag indicates drawing priority:
        #
        # gl.GL_STREAM_DRAW  =  most frequent update
        # gl.GL_DYNAMIC_DRAW =  update
        # gl.GL_STATIC_DRAW  =  less frequent update

        self.achdgbuf = GLBuffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.aclatbuf = GLBuffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.aclonbuf = GLBuffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.acaltbuf = GLBuffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.actasbuf = GLBuffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.accolorbuf = GLBuffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.aclblbuf = GLBuffer(MAX_NAIRCRAFT * 24, usage=gl.GL_STREAM_DRAW)
        self.confcpabuf = GLBuffer(MAX_NCONFLICTS * 16, usage=gl.GL_STREAM_DRAW)
        self.asasnbuf = GLBuffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.asasebuf = GLBuffer(MAX_NAIRCRAFT * 4, usage=gl.GL_STREAM_DRAW)
        self.routewplatbuf = GLBuffer(ROUTE_SIZE * 4, usage=gl.GL_DYNAMIC_DRAW)
        self.routewplonbuf = GLBuffer(ROUTE_SIZE * 4, usage=gl.GL_DYNAMIC_DRAW)
        self.routelblbuf = GLBuffer(ROUTE_SIZE * 2*12, usage=gl.GL_DYNAMIC_DRAW)

        self.custwplblbuf = GLBuffer(CUSTWP_SIZE * 10, usage=gl.GL_STATIC_DRAW)

        # ------- Map ------------------------------------
        texcoords = np.array([(1, 3), (1, 0), (0, 0), (0, 3)], dtype=np.float32)
        self.map = Rectangle(1080.0, 180.0, True, texcoords=texcoords, texture=map_texname)

        # ------- Coastlines -----------------------------
        self.coastlines = VertexAttributeObject(gl.GL_LINES, vertex=self.coastvertices, color=palette.coastlines)
        self.vcount_coast = len(self.coastvertices)
        del self.coastvertices

        # ------- Airport graphics -----------------------
        self.runways = VertexAttributeObject(gl.GL_TRIANGLES, vertex=self.vbuf_runways, color=palette.runways)
        self.thresholds = VertexAttributeObject(gl.GL_TRIANGLES, vertex=self.vbuf_rwythr, color=palette.thresholds)
        self.taxiways = VertexAttributeObject(gl.GL_TRIANGLES, vertex=self.vbuf_asphalt, color=palette.taxiways)
        self.pavement = VertexAttributeObject(gl.GL_TRIANGLES, vertex=self.vbuf_concrete, color=palette.pavement)

        # Polygon preview object
        self.polyprev = VertexAttributeObject(gl.GL_LINE_LOOP, vertex=POLYPREV_SIZE * 8, color=palette.previewpoly, usage=gl.GL_DYNAMIC_DRAW)

        # Fixed polygons
        self.allpolys = VertexAttributeObject(gl.GL_LINES, vertex=POLY_SIZE * 16, color=POLY_SIZE * 8)
        self.allpfill = VertexAttributeObject(gl.GL_TRIANGLES, vertex=POLY_SIZE * 24, color=np.append(palette.polys, 50))

        # ------- SSD object -----------------------------
        self.ssd = VertexAttributeObject(gl.GL_POINTS, shader_type='ssd')
        self.ssd.selssd.bind(MAX_NAIRCRAFT, datatype=gl.GL_UNSIGNED_BYTE, instance_divisor=1)
        self.ssd.set_attribs(lat0=self.aclatbuf, lon0=self.aclonbuf,
                             alt0=self.acaltbuf, tas0=self.actasbuf,
                             trk0=self.achdgbuf, asasn=self.asasnbuf,
                             asase=self.asasebuf, instance_divisor=1)
        self.ssd.set_attribs(lat1=self.aclatbuf, lon1=self.aclonbuf,
                             alt1=self.acaltbuf, tas1=self.actasbuf,
                             trk1=self.achdgbuf)

        # ------- Protected Zone -------------------------
        self.protectedzone = Circle(radius=2.5 * nm)
        self.protectedzone.set_attribs(lat=self.aclatbuf, lon=self.aclonbuf,
                                       color=self.accolorbuf, instance_divisor=1)

        # ------- A/C symbol -----------------------------
        acvertices = np.array([(0.0, 0.5 * ac_size), (-0.5 * ac_size, -0.5 * ac_size), (0.0, -0.25 * ac_size), (0.5 * ac_size, -0.5 * ac_size)], dtype=np.float32)
        self.ac_symbol = VertexAttributeObject(gl.GL_TRIANGLE_FAN, vertex=acvertices)
        self.ac_symbol.set_attribs(lat=self.aclatbuf, lon=self.aclonbuf,
                                   color=self.accolorbuf, orientation=self.achdgbuf, instance_divisor=1)
        self.aclabels = self.font.prepare_text_instanced(self.aclblbuf, (8, 3), self.aclatbuf, self.aclonbuf, self.accolorbuf, char_size=text_size, vertex_offset=(ac_size, -0.5 * ac_size))

        # ------- Conflict CPA lines ---------------------
        self.cpalines = VertexAttributeObject(gl.GL_LINES, vertex=self.confcpabuf, color=palette.conflict)

        # ------- Aircraft Route -------------------------
        self.route = VertexAttributeObject(gl.GL_LINES, vertex=ROUTE_SIZE * 8, color=palette.route, usage=gl.GL_DYNAMIC_DRAW)
        self.routelbl = self.font.prepare_text_instanced(self.routelblbuf, (12, 2), self.routewplatbuf, self.routewplonbuf, char_size=text_size, vertex_offset=(wpt_size, 0.5 * wpt_size))
        self.routelbl.color.bind(palette.route)
        rwptvertices = np.array([(-0.2 * wpt_size, -0.2 * wpt_size),
                                 ( 0.0,            -0.8 * wpt_size),
                                 ( 0.2 * wpt_size, -0.2 * wpt_size),
                                 ( 0.8 * wpt_size,  0.0),
                                 ( 0.2 * wpt_size,  0.2 * wpt_size),
                                 ( 0.0,             0.8 * wpt_size),
                                 (-0.2 * wpt_size,  0.2 * wpt_size),
                                 (-0.8 * wpt_size,  0.0)], dtype=np.float32)
        self.rwaypoints = VertexAttributeObject(gl.GL_LINE_LOOP, vertex=rwptvertices, color=palette.route)
        self.rwaypoints.set_attribs(lat=self.routewplatbuf, lon=self.routewplonbuf, instance_divisor=1)

        # --------Aircraft Trails------------------------------------------------
        self.traillines = VertexAttributeObject(gl.GL_LINES, vertex=TRAILS_SIZE * 16, color=palette.trails)

        # ------- Waypoints ------------------------------
        wptvertices = np.array([(0.0, 0.5 * wpt_size), (-0.5 * wpt_size, -0.5 * wpt_size), (0.5 * wpt_size, -0.5 * wpt_size)], dtype=np.float32)  # a triangle
        self.nwaypoints = len(bs.navdb.wplat)
        self.waypoints = VertexAttributeObject(gl.GL_LINE_LOOP, vertex=wptvertices, color=palette.wptsymbol, n_instances=self.nwaypoints)
        # Sort based on id string length
        llid = sorted(zip(bs.navdb.wpid, bs.navdb.wplat, bs.navdb.wplon), key=lambda i: len(i[0]) > 3)
        wpidlst, wplat, wplon = zip(*llid)
        self.waypoints.set_attribs(lat=np.array(wplat, dtype=np.float32), lon=np.array(wplon, dtype=np.float32), instance_divisor=1)
        wptids = ''
        self.nnavaids = 0
        for wptid in wpidlst:
            if len(wptid) <= 3:
                self.nnavaids += 1
            wptids += wptid[:5].ljust(5)
        npwpids = np.array(wptids, dtype=np.string_)
        self.wptlabels = self.font.prepare_text_instanced(npwpids, (5, 1), self.waypoints.lat, self.waypoints.lon, char_size=text_size, vertex_offset=(wpt_size, 0.5 * wpt_size))
        self.wptlabels.color.bind(palette.wptlabel)
        del wptids
        self.customwp = VertexAttributeObject(gl.GL_LINE_LOOP, vertex=self.waypoints.vertex, color=palette.wptsymbol)
        self.customwp.set_attribs(lat=CUSTWP_SIZE * 4, lon=CUSTWP_SIZE * 4, instance_divisor=1)
        self.customwplbl = self.font.prepare_text_instanced(self.custwplblbuf, (10, 1), self.customwp.lat, self.customwp.lon, char_size=text_size, vertex_offset=(wpt_size, 0.5 * wpt_size))
        self.customwplbl.color.bind(palette.wptlabel)
        # ------- Airports -------------------------------
        aptvertices = np.array([(-0.5 * apt_size, -0.5 * apt_size), (0.5 * apt_size, -0.5 * apt_size), (0.5 * apt_size, 0.5 * apt_size), (-0.5 * apt_size, 0.5 * apt_size)], dtype=np.float32)  # a square
        self.nairports = len(bs.navdb.aptlat)
        self.airports = VertexAttributeObject(gl.GL_LINE_LOOP, vertex=aptvertices, color=palette.aptsymbol, n_instances=self.nairports)
        indices = bs.navdb.aptype.argsort()
        aplat   = np.array(bs.navdb.aptlat[indices], dtype=np.float32)
        aplon   = np.array(bs.navdb.aptlon[indices], dtype=np.float32)
        aptypes = bs.navdb.aptype[indices]
        apnames = np.array(bs.navdb.aptid)
        apnames = apnames[indices]
        # The number of large, large+med, and large+med+small airports
        self.nairports = [aptypes.searchsorted(2), aptypes.searchsorted(3), self.nairports]

        self.airports.set_attribs(lat=aplat, lon=aplon, instance_divisor=1)
        aptids = ''
        for aptid in apnames:
            aptids += aptid.ljust(4)
        self.aptlabels = self.font.prepare_text_instanced(np.array(aptids, dtype=np.string_), (4, 1), self.airports.lat, self.airports.lon, char_size=text_size, vertex_offset=(apt_size, 0.5 * apt_size))
        self.aptlabels.color.bind(palette.aptlabel)
        del aptids

        # Unbind VAO, VBO
        VertexAttributeObject.unbind_all()

        # Set initial values for the global uniforms
        self.shaderset.set_wrap(self.wraplon, self.wrapdir)
        self.shaderset.set_pan_and_zoom(self.panlat, self.panlon, self.zoom)

        # Clean up memory
        del self.vbuf_asphalt, self.vbuf_concrete, self.vbuf_runways, self.vbuf_rwythr

        self.initialized = True

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""

        # First check for supported GL version
        gl_version = float(gl.glGetString(gl.GL_VERSION)[:3])
        if gl_version < 3.3:
            print(('OpenGL context created with GL version %.1f' % gl_version))
            qCritical("""Your system reports that it supports OpenGL up to version %.1f. The minimum requirement for BlueSky is OpenGL 3.3.
                Generally, AMD/ATI/nVidia cards from 2008 and newer support OpenGL 3.3, and Intel integrated graphics from the Haswell
                generation and newer. If you think your graphics system should be able to support GL>=3.3 please open an issue report
                on the BlueSky Github page (https://github.com/ProfHoekstra/bluesky/issues)""" % gl_version)
            return

        # background color
        gl.glClearColor(*(palette.background + (0,)))
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        try:
            self.shaderset.load_shaders()

            self.color_shader = self.shaderset['normal']

            # Compile shaders and link texture shader program
            self.texture_shader = self.shaderset['textured']

            # Compile shaders and link text shader program
            self.text_shader = self.shaderset['text']

            self.ssd_shader = self.shaderset['ssd']

        except RuntimeError as e:
            print('Error compiling shaders in radarwidget: ' + e.args[0])
            qCritical('Error compiling shaders in radarwidget: ' + e.args[0])
            return

        # create all vertex array objects
        # try:
        self.create_objects()
        # except Exception as e:
        #     print('Error while creating RadarWidget objects: ' + e.args[0])

    def paintGL(self):
        """Paint the scene."""
        # pass if the framebuffer isn't complete yet or if not initialized
        if not (gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE and self.initialized and self.isVisible()):
            return

        # Get data for active node
        actdata = bs.net.get_nodedata()

        # Set the viewport and clear the framebuffer
        gl.glViewport(*self.viewport)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Send the (possibly) updated global uniforms to the buffer
        self.shaderset.set_vertex_scale_type(VERTEX_IS_LATLON)

        # --- DRAW THE MAP AND COASTLINES ---------------------------------------------
        # Map and coastlines: don't wrap around in the shader
        self.shaderset.enable_wrap(False)

        if actdata.show_map:
            # Select the texture shader
            # self.texture_shader.use()

            # Draw map texture
            # self.map_texture.bind()
            self.map.draw()

        # Select the non-textured shader
        # self.color_shader.use()

        # Draw coastlines
        if actdata.show_coast:
            if self.wrapdir == 0:
                # Normal case, no wrap around
                self.coastlines.draw(first_vertex=0, vertex_count=self.vcount_coast)
            else:
                self.coastlines.bind()
                wrapindex = np.uint32(self.coastindices[int(self.wraplon) + 180])
                if self.wrapdir == 1:
                    gl.glVertexAttrib1f(self.coastlines.lon.loc, 360.0)
                    self.coastlines.draw(first_vertex=0, vertex_count=wrapindex)
                    gl.glVertexAttrib1f(self.coastlines.lon.loc, 0.0)
                    self.coastlines.draw(first_vertex=wrapindex, vertex_count=self.vcount_coast - wrapindex)
                else:
                    gl.glVertexAttrib1f(self.coastlines.lon.loc, -360.0)
                    self.coastlines.draw(first_vertex=wrapindex, vertex_count=self.vcount_coast - wrapindex)
                    gl.glVertexAttrib1f(self.coastlines.lon.loc, 0.0)
                    self.coastlines.draw(first_vertex=0, vertex_count=wrapindex)

        # --- DRAW PREVIEW SHAPE (WHEN AVAILABLE) -----------------------------
        self.polyprev.draw()

        # --- DRAW CUSTOM SHAPES (WHEN AVAILABLE) -----------------------------
        if actdata.show_poly > 0:
            self.allpolys.draw()
            if actdata.show_poly > 1:
                self.allpfill.draw()

        # --- DRAW THE SELECTED AIRCRAFT ROUTE (WHEN AVAILABLE) ---------------
        if actdata.show_traf:
            self.route.draw()
            self.cpalines.draw()
            self.traillines.draw()

        # --- DRAW AIRPORT DETAILS (RUNWAYS, TAXIWAYS, PAVEMENTS) -------------
        self.runways.draw()
        self.thresholds.draw()

        if self.zoom >= 1.0:
            for idx in self.apt_inrange:
                self.taxiways.draw(first_vertex=idx[0], vertex_count=idx[1])
                self.pavement.draw(first_vertex=idx[2], vertex_count=idx[3])

        # --- DRAW THE INSTANCED AIRCRAFT SHAPES ------------------------------
        # update wrap longitude and direction for the instanced objects
        self.shaderset.enable_wrap(True)

        # PZ circles only when they are bigger than the A/C symbols
        if self.naircraft > 0 and actdata.show_traf and actdata.show_pz and self.zoom >= 0.15:
            self.shaderset.set_vertex_scale_type(VERTEX_IS_METERS)
            self.protectedzone.draw(n_instances=self.naircraft)

        self.shaderset.set_vertex_scale_type(VERTEX_IS_SCREEN)

        # Draw traffic symbols
        if self.naircraft > 0 and actdata.show_traf:
            if self.routelbl.n_instances:
                self.rwaypoints.draw(n_instances=self.routelbl.n_instances)
            self.ac_symbol.draw(n_instances=self.naircraft)

        if self.zoom >= 0.5 and actdata.show_apt == 1 or actdata.show_apt == 2:
            nairports = self.nairports[2]
        elif self.zoom  >= 0.25 and actdata.show_apt == 1 or actdata.show_apt == 3:
            nairports = self.nairports[1]
        else:
            nairports = self.nairports[0]

        if self.zoom >= 3 and actdata.show_wpt == 1 or actdata.show_wpt == 2:
            nwaypoints = self.nwaypoints
        else:
            nwaypoints = self.nnavaids

        # Draw waypoint symbols
        if actdata.show_wpt:
            self.waypoints.draw(n_instances=nwaypoints)
            if self.ncustwpts > 0:
                self.customwp.draw(n_instances=self.ncustwpts)

        # Draw airport symbols
        if actdata.show_apt:
            self.airports.draw(n_instances=nairports)

        # Text rendering
        self.text_shader.use()
        self.font.use()

        if actdata.show_apt:
            self.font.set_char_size(self.aptlabels.char_size)
            self.font.set_block_size(self.aptlabels.block_size)
            self.aptlabels.draw(n_instances=nairports)
        if actdata.show_wpt:
            self.font.set_char_size(self.wptlabels.char_size)
            self.font.set_block_size(self.wptlabels.block_size)
            self.wptlabels.draw(n_instances=nwaypoints)
            if self.ncustwpts > 0:
                self.font.set_char_size(self.customwplbl.char_size)
                self.font.set_block_size(self.customwplbl.block_size)
                self.customwplbl.draw(n_instances=self.ncustwpts)

        if actdata.show_traf and self.route.vertex_count > 1:
            self.font.set_char_size(self.routelbl.char_size)
            self.font.set_block_size(self.routelbl.block_size)
            self.routelbl.draw()

        if self.naircraft > 0 and actdata.show_traf and actdata.show_lbl:
            self.font.set_char_size(self.aclabels.char_size)
            self.font.set_block_size(self.aclabels.block_size)
            self.aclabels.draw(n_instances=self.naircraft)

        # SSD
        if actdata.ssd_all or actdata.ssd_conflicts or len(actdata.ssd_ownship) > 0:
            self.ssd_shader.use()
            gl.glUniform3f(self.ssd_shader.Vlimits.loc, self.asas_vmin ** 2, self.asas_vmax ** 2, self.asas_vmax)
            gl.glUniform1i(self.ssd_shader.n_ac.loc, self.naircraft)
            self.ssd.draw(vertex_count=self.naircraft, n_instances=self.naircraft)

        # Unbind everything
        # VertexAttributeObject.unbind_all()
        # gl.glUseProgram(0)

    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport."""
        if not self.initialized:
            return

        # update the window size
        # Qt5 supports getting the device pixel ratio, which can be > 1 for HiDPI displays such as Mac Retina screens
        pixel_ratio = self.devicePixelRatio()

        # Calculate zoom so that the window resize doesn't affect the scale, but only enlarges or shrinks the view
        zoom   = float(self.width) / float(width) * pixel_ratio
        origin = (width / 2, height / 2)

        # Update width, height, and aspect ratio
        self.width, self.height = width // pixel_ratio, height // pixel_ratio
        self.ar = float(width) / max(1, float(height))
        self.shaderset.set_win_width_height(self.width, self.height)
        self.viewport = (0, 0, width, height)

        # Update zoom
        self.panzoom(zoom=zoom, origin=origin)

    def on_simstream_received(self, streamname, data, sender_id):
        if streamname == b'ACDATA':
            self.acdata = ACDataEvent(data)
            self.update_aircraft_data(self.acdata)
        elif streamname[:9] == b'ROUTEDATA':
            self.routedata = RouteDataEvent(data)
            self.update_route_data(self.routedata)

    def update_route_data(self, data):
        if not self.initialized:
            return
        self.makeCurrent()
        actdata = bs.net.get_nodedata()

        self.route_acid = data.acid
        if data.acid != "" and len(data.wplat) > 0:
            nsegments = len(data.wplat)
            data.iactwp = min(max(0, data.iactwp), nsegments - 1)
            self.routelbl.n_instances = nsegments
            self.route.set_vertex_count(2 * nsegments)
            routedata = np.empty(4 * nsegments, dtype=np.float32)
            routedata[0:4] = [data.aclat, data.aclon,
                              data.wplat[data.iactwp], data.wplon[data.iactwp]]

            routedata[4::4] = data.wplat[:-1]
            routedata[5::4] = data.wplon[:-1]
            routedata[6::4] = data.wplat[1:]
            routedata[7::4] = data.wplon[1:]

            self.route.update(vertex=routedata)
            self.routewplatbuf.update(np.array(data.wplat, dtype=np.float32))
            self.routewplonbuf.update(np.array(data.wplon, dtype=np.float32))
            wpname = ''
            for wp, alt, spd in zip(data.wpname, data.wpalt, data.wpspd):
                if alt < 0. and spd < 0.:
                    txt = wp[:12].ljust(24) # No second line
                else:
                    txt = wp[:12].ljust(12) # Two lines
                    if alt < 0:
                        txt += "-----/"
                    elif alt > actdata.translvl:
                        FL = int(round((alt / (100. * ft))))
                        txt += "FL%03d/" % FL
                    else:
                        txt += "%05d/" % int(round(alt / ft))

                    # Speed
                    if spd < 0:
                        txt += "--- "
                    elif spd>2.0:
                        txt += "%03d" % int(round(spd / kts))
                    else:
                        txt += "M{:.2f}".format(spd) # Mach number

                wpname += txt.ljust(24) # Fill out with spaces
            self.routelblbuf.update(np.array(wpname.encode('ascii', 'ignore')))
        else:
            self.route.set_vertex_count(0)
            self.routelbl.n_instances = 0

    def update_aircraft_data(self, data):
        if not self.initialized:
            return

        self.makeCurrent()
        actdata = bs.net.get_nodedata()
        if actdata.filteralt:
            idx = np.where((data.alt >= actdata.filteralt[0]) * (data.alt <= actdata.filteralt[1]))
            data.lat = data.lat[idx]
            data.lon = data.lon[idx]
            data.trk = data.trk[idx]
            data.alt = data.alt[idx]
            data.tas = data.tas[idx]
            data.vs  = data.vs[idx]
        self.naircraft = len(data.lat)
        actdata.translvl = data.translvl
        self.asas_vmin = data.vmin
        self.asas_vmax = data.vmax
        if self.naircraft == 0:
            self.cpalines.set_vertex_count(0)
        else:
            # Update data in GPU buffers
            self.aclatbuf.update(np.array(data.lat, dtype=np.float32))
            self.aclonbuf.update(np.array(data.lon, dtype=np.float32))
            self.achdgbuf.update(np.array(data.trk, dtype=np.float32))
            self.acaltbuf.update(np.array(data.alt, dtype=np.float32))
            self.actasbuf.update(np.array(data.tas, dtype=np.float32))
            self.asasnbuf.update(np.array(data.asasn, dtype=np.float32))
            self.asasebuf.update(np.array(data.asase, dtype=np.float32))

            # CPA lines to indicate conflicts
            ncpalines = np.count_nonzero(data.inconf)

            cpalines  = np.zeros(4 * ncpalines, dtype=np.float32)
            self.cpalines.set_vertex_count(2 * ncpalines)

            # Labels and colors
            rawlabel = ''
            color = np.empty((min(self.naircraft, MAX_NAIRCRAFT), 4), dtype=np.uint8)
            selssd = np.zeros(self.naircraft, dtype=np.uint8)
            confidx = 0

            zdata = zip(data.id, data.ingroup, data.inconf, data.tcpamax, data.trk, data.gs,
                        data.cas, data.vs, data.alt, data.lat, data.lon)
            for i, (acid, ingroup, inconf, tcpa, trk, gs, cas, vs, alt, lat, lon) in enumerate(zdata):
                if i >= MAX_NAIRCRAFT:
                    break

                # Make label: 3 lines of 8 characters per aircraft
                if actdata.show_lbl >= 1:
                    rawlabel += '%-8s' % acid[:8]
                    if actdata.show_lbl == 2:
                        if alt <= data.translvl:
                            rawlabel += '%-5d' % int(alt / ft  + 0.5)
                        else:
                            rawlabel += 'FL%03d' % int(alt / ft / 100. + 0.5)
                        vsarrow = 30 if vs > 0.25 else 31 if vs < -0.25 else 32
                        rawlabel += '%1s  %-8d' % (chr(vsarrow), int(cas / kts + 0.5))
                    else:
                        rawlabel += 16 * ' '

                if inconf:
                    if actdata.ssd_conflicts:
                        selssd[i] = 255
                    color[i, :] = palette.conflict + (255,)
                    lat1, lon1 = geo.qdrpos(lat, lon, trk, tcpa * gs / nm)
                    cpalines[4 * confidx : 4 * confidx + 4] = [lat, lon, lat1, lon1]
                    confidx += 1
                else:
                    # Get custom color if available, else default
                    rgb = palette.aircraft
                    if ingroup:
                        for groupmask, groupcolor in actdata.custgrclr.items():
                            if ingroup & groupmask:
                                rgb = groupcolor
                                break
                    rgb = actdata.custacclr.get(acid, rgb)
                    color[i, :] = tuple(rgb) + (255,)

                #  Check if aircraft is selected to show SSD
                if actdata.ssd_all or acid in actdata.ssd_ownship:
                    selssd[i] = 255

            if len(actdata.ssd_ownship) > 0 or actdata.ssd_conflicts or actdata.ssd_all:
                self.ssd.selssd.buf.update(selssd)

            self.confcpabuf.update(cpalines)
            self.accolorbuf.update(color)
            self.aclblbuf.update(np.array(rawlabel.encode('utf8'), dtype=np.string_))

            # If there is a visible route, update the start position
            if self.route_acid != "":
                if self.route_acid in data.id:
                    idx = data.id.index(self.route_acid)
                    self.route.vertex.update(
                                  np.array([data.lat[idx], data.lon[idx]], dtype=np.float32))

            # Update trails database with new lines
            if data.swtrails:
                actdata.traillat0.extend(data.traillat0)
                actdata.traillon0.extend(data.traillon0)
                actdata.traillat1.extend(data.traillat1)
                actdata.traillon1.extend(data.traillon1)
                self.traillines.update(vertex=np.array(
                    list(zip(actdata.traillat0, actdata.traillon0,
                             actdata.traillat1, actdata.traillon1)) +
                    list(zip(data.traillastlat, data.traillastlon,
                             list(data.lat), list(data.lon))),
                    dtype=np.float32))

            else:
                actdata.traillat0 = []
                actdata.traillon0 = []
                actdata.traillat1 = []
                actdata.traillon1 = []

                self.traillines.set_vertex_count(0)

    def cmdline_stacked(self, cmd, args):
        if cmd in ['AREA', 'BOX', 'POLY', 'POLYGON', 'CIRCLE', 'LINE','POLYLINE']:
            self.polyprev.set_vertex_count(0)

    def previewpoly(self, shape_type, data_in=None):
        if not self.initialized:
            return
        self.makeCurrent()

        if shape_type is None:
            self.polyprev.set_vertex_count(0)
            return
        if shape_type in ['BOX', 'AREA']:
            # For a box (an area is a box) we need to add two additional corners
            data = np.zeros(8, dtype=np.float32)
            data[0:2] = data_in[0:2]
            data[2:4] = data_in[2], data_in[1]
            data[4:6] = data_in[2:4]
            data[6:8] = data_in[0], data_in[3]
        else:
            data = np.array(data_in, dtype=np.float32)

        if shape_type[-4:] == 'LINE':
            self.polyprev.set_primitive_type(gl.GL_LINE_STRIP)
        else:
            self.polyprev.set_primitive_type(gl.GL_LINE_LOOP)

        self.polyprev.update(vertex=data)

    def pixelCoordsToGLxy(self, x, y):
        """Convert screen pixel coordinates to GL projection coordinates (x, y range -1 -- 1)
        """
        # GL coordinates (x, y range -1 -- 1)
        glx = (float(2.0 * x) / self.width  - 1.0)
        gly = -(float(2.0 * y) / self.height - 1.0)
        return glx, gly

    def pixelCoordsToLatLon(self, x, y):
        """Convert screen pixel coordinates to lat/lon coordinates
        """
        glx, gly = self.pixelCoordsToGLxy(x, y)

        # glxy   = zoom * (latlon - pan)
        # latlon = pan + glxy / zoom
        lat = self.panlat + gly / (self.zoom * self.ar)
        lon = self.panlon + glx / (self.zoom * self.flat_earth)
        return lat, lon

    def panzoom(self, pan=None, zoom=None, origin=None, absolute=False):
        if not self.initialized:
            return False

        if pan:
            # Absolute pan operation
            if absolute:
                self.panlat = pan[0]
                self.panlon = pan[1]
            # Relative pan operation
            else:
                self.panlat += pan[0]
                self.panlon += pan[1]

            # Don't pan further than the poles in y-direction
            self.panlat = min(max(self.panlat, -90.0 + 1.0 /
                  (self.zoom * self.ar)), 90.0 - 1.0 / (self.zoom * self.ar))

            # Update flat-earth factor and possibly zoom in case of very wide windows (> 2:1)
            self.flat_earth = np.cos(np.deg2rad(self.panlat))
            self.zoom = max(self.zoom, 1.0 / (180.0 * self.flat_earth))

        if zoom:
            if absolute:
                # Limit zoom extents in x-direction to [-180:180], and in y-direction to [-90:90]
                self.zoom = max(zoom, 1.0 / min(90.0 * self.ar, 180.0 * self.flat_earth))
            else:
                prevzoom = self.zoom
                glx, gly = self.pixelCoordsToGLxy(*origin) if origin else (0,0)
                self.zoom *= zoom

                # Limit zoom extents in x-direction to [-180:180], and in y-direction to [-90:90]
                self.zoom = max(self.zoom, 1.0 / min(90.0 * self.ar, 180.0 * self.flat_earth))

                # Correct pan so that zoom actions are around the mouse position, not around 0, 0
                # glxy / zoom1 - pan1 = glxy / zoom2 - pan2
                # pan2 = pan1 + glxy (1/zoom2 - 1/zoom1)
                self.panlon = self.panlon - glx * (1.0 / self.zoom - 1.0 / prevzoom) / self.flat_earth
                self.panlat = self.panlat - gly * (1.0 / self.zoom - 1.0 / prevzoom) / self.ar

            # Don't pan further than the poles in y-direction
            self.panlat = min(max(self.panlat, -90.0 + 1.0 / (self.zoom * self.ar)), 90.0 - 1.0 / (self.zoom * self.ar))

            # Update flat-earth factor
            self.flat_earth = np.cos(np.deg2rad(self.panlat))

        if self.zoom >= 1.0:
            # Airports may be visible when zoom > 1: in this case, update the list of indicates
            # of airports that need to be drawn
            ll_range = max(1.5 / self.zoom, 1.0)
            indices = np.logical_and(np.abs(self.apt_ctrlat - self.panlat) <= ll_range, np.abs(self.apt_ctrlon - self.panlon) <= ll_range)
            self.apt_inrange = self.apt_indices[indices]

        # Check for necessity wrap-around in x-direction
        self.wraplon  = -999.9
        self.wrapdir  = 0
        if self.panlon + 1.0 / (self.zoom * self.flat_earth) < -180.0:
            # The left edge of the map has passed the right edge of the screen: we can just change the pan position
            self.panlon += 360.0
        elif self.panlon - 1.0 / (self.zoom * self.flat_earth) < -180.0:
            # The left edge of the map has passed the left edge of the screen: we need to wrap around to the left
            self.wraplon = float(np.ceil(360.0 + self.panlon - 1.0 / (self.zoom * self.flat_earth)))
            self.wrapdir = -1
        elif self.panlon - 1.0 / (self.zoom * self.flat_earth) > 180.0:
            # The right edge of the map has passed the left edge of the screen: we can just change the pan position
            self.panlon -= 360.0
        elif self.panlon + 1.0 / (self.zoom * self.flat_earth) > 180.0:
            # The right edge of the map has passed the right edge of the screen: we need to wrap around to the right
            self.wraplon = float(np.floor(-360.0 + self.panlon + 1.0 / (self.zoom * self.flat_earth)))
            self.wrapdir = 1

        self.shaderset.set_wrap(self.wraplon, self.wrapdir)

        # update pan and zoom on GPU for all shaders
        self.shaderset.set_pan_and_zoom(self.panlat, self.panlon, self.zoom)
        # Update pan and zoom in centralized nodedata
        bs.net.get_nodedata().panzoom((self.panlat, self.panlon), self.zoom)

        return True

    def event(self, event):
        ''' Event handling for input events. '''
        if event.type() == QEvent.Wheel:
            # For mice we zoom with control/command and the scrolwheel
            if event.modifiers() & Qt.ControlModifier:
                origin = (event.pos().x(), event.pos().y())
                zoom = 1.0
                try:
                    if event.pixelDelta():
                        # High resolution scroll
                        zoom *= (1.0 + 0.01 * event.pixelDelta().y())
                    else:
                        # Low resolution scroll
                        zoom *= (1.0 + 0.001 * event.angleDelta().y())
                except AttributeError:
                    zoom *= (1.0 + 0.001 * event.delta())
                self.panzoomchanged = True
                return self.panzoom(zoom=zoom, origin=origin)

            # For touchpad scroll (2D) is used for panning
            else:
                try:
                    dlat = 0.01 * event.pixelDelta().y() / (self.zoom * self.ar)
                    dlon = -0.01 * event.pixelDelta().x() / (self.zoom * self.flat_earth)
                    self.panzoomchanged = True
                    return self.panzoom(pan=(dlat, dlon))
                except AttributeError:
                    pass

        # For touchpad, pinch gesture is used for zoom
        elif event.type() == QEvent.Gesture:
            pan = zoom = None
            dlat = dlon = 0.0
            for g in event.gestures():
                if g.gestureType() == Qt.PinchGesture:
                    zoom = g.scaleFactor() * (zoom or 1.0)
                    if CORRECT_PINCH:
                        zoom /= g.lastScaleFactor()
                elif g.gestureType() == Qt.PanGesture:
                    if abs(g.delta().y() + g.delta().x()) > 1e-1:
                        dlat += 0.005 * g.delta().y() / (self.zoom * self.ar)
                        dlon -= 0.005 * g.delta().x() / (self.zoom * self.flat_earth)
                        pan = (dlat, dlon)
            if pan is not None or zoom is not None:
                self.panzoomchanged = True
                return self.panzoom(pan, zoom, self.mousepos)

        elif event.type() == QEvent.MouseButtonPress and event.button() & Qt.LeftButton:
            self.mousedragged = False
            # For mice we pan with control/command and mouse movement.
            # Mouse button press marks the beginning of a pan
            self.prevmousepos = (event.x(), event.y())

        elif event.type() == QEvent.MouseButtonRelease and \
             event.button() & Qt.LeftButton and not self.mousedragged:
            lat, lon = self.pixelCoordsToLatLon(event.x(), event.y())
            # TODO: acdata en routedata
            tostack, tocmdline = radarclick(console.get_cmdline(), lat, lon,
                                            self.acdata, self.routedata)
            if '\n' not in tocmdline:
                console.append_cmdline(tocmdline)
            if tostack:
                console.stack(tostack)

        elif event.type() == QEvent.MouseMove:
            self.mousedragged = True
            self.mousepos = (event.x(), event.y())
            if event.buttons() & Qt.LeftButton:
                dlat = 0.003 * (event.y() - self.prevmousepos[1]) / (self.zoom * self.ar)
                dlon = 0.003 * (self.prevmousepos[0] - event.x()) / (self.zoom * self.flat_earth)
                self.prevmousepos = (event.x(), event.y())
                self.panzoomchanged = True
                return self.panzoom(pan=(dlat, dlon))

        # Update pan/zoom to simulation thread only when the pan/zoom gesture is finished
        elif (event.type() == QEvent.MouseButtonRelease or
              event.type() == QEvent.TouchEnd) and self.panzoomchanged:
            self.panzoomchanged = False
            bs.net.send_event(b'PANZOOM', dict(pan=(self.panlat, self.panlon),
                                           zoom=self.zoom, ar=self.ar, absolute=True))

        # If this is a mouse move event, check if we are updating a preview poly
        if self.mousepos != self.prevmousepos:
            cmd = console.get_cmd()
            nargs = len(console.get_args())
            if cmd in ['AREA', 'BOX', 'POLY','POLYLINE',
                       'POLYALT', 'POLYGON', 'CIRCLE', 'LINE'] and nargs >= 2:
                self.prevmousepos = self.mousepos
                try:
                    # get the largest even number of points
                    start = 0 if cmd == 'AREA' else 3 if cmd == 'POLYALT' else 1
                    end = ((nargs - start) // 2) * 2 + start
                    data = [float(v) for v in console.get_args()[start:end]]
                    data += self.pixelCoordsToLatLon(*self.mousepos)
                    self.previewpoly(cmd, data)

                except ValueError:
                    pass

        # For all other events call base class event handling
        return super(RadarWidget, self).event(event)
