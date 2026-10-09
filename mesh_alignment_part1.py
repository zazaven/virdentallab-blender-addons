# Acknowledgement:
# Landmark picking (placeSeed), the landmark label overlay (drawTextCallback)
# and the landmark/vertex-selection initial alignment in this file are adapted
# from the "Iterative Closest Point (ICP) Registration" add-on v3.2 by
# 3D OPERATORS (https://superhivemarket.com/products/icp-iterative-closest-point-registration-addon),
# distributed under GPL-compatible terms. Modified by virdentallab.

bl_info = {
    "name": "Mesh Alignment Pro",
    "author": "virdentallab",
    "version": (1, 0, 1),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Mesh Align",
    "description": "Mesh alignment with paint selection, landmarks, and Open3D ICP",
    "doc_url": "https://github.com/zazaven/virdentallab-blender-addons",
    "category": "Mesh",
}

import bpy
import numpy as np
import math as mt
import mathutils as mu
import bmesh
import copy
import blf
import time
import sys
import os
import subprocess
from bpy.types import Panel, Operator, PropertyGroup
from bpy.props import (PointerProperty, BoolProperty, IntProperty, FloatProperty, 
                      EnumProperty, StringProperty, FloatVectorProperty)
from bpy_extras import view3d_utils
from mathutils import Matrix, Vector, kdtree

# Add user site-packages to path for Open3D
user_site = os.path.expanduser(rf"~\AppData\Roaming\Python\Python{sys.version_info.major}{sys.version_info.minor}\site-packages")
if user_site not in sys.path:
    sys.path.insert(0, user_site)

# Try to import visualization modules
try:
    import gpu
    from gpu_extras.batch import batch_for_shader
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False
    print("GPU module not available - visualization features will be limited")

# ===================== DEPENDENCY CHECK =====================

def check_open3d_installed():
    """Check if Open3D is installed and available"""
    try:
        import open3d as o3d
        return True, o3d.__version__
    except ImportError as e:
        error_msg = str(e)
        # Check for common missing dependencies
        if "markupsafe" in error_msg.lower():
            print("Open3D dependency missing: markupsafe")
        elif "jinja2" in error_msg.lower():
            print("Open3D dependency missing: jinja2")
        else:
            print(f"Open3D import error: {error_msg}")
        return False, None

def get_python_executable():
    """Get the Python executable path for Blender"""
    return sys.executable


# ===================== HELPER UTILITIES =====================

def collect_world_vertices(mesh_obj, prefer_selected=True, min_selected=4):
    """Collect Nx3 world-space vertices from a mesh object.

    If prefer_selected=True and the mesh has at least min_selected selected
    vertices, returns those; otherwise returns all vertices.

    Returns: (np.ndarray float32 of shape (N,3), used_selection: bool)
    """
    if prefer_selected:
        selected = [v for v in mesh_obj.data.vertices if v.select]
        if len(selected) >= min_selected:
            verts = np.array(
                [mesh_obj.matrix_world @ v.co for v in selected],
                dtype=np.float32,
            )
            return verts, True

    verts = np.array(
        [mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices],
        dtype=np.float32,
    )
    return verts, False


def apply_world_transform(mesh_obj, transform_4x4):
    """Apply a 4x4 transform (mathutils.Matrix or list/np.ndarray) to a mesh
    object's world matrix in-place."""
    if not isinstance(transform_4x4, Matrix):
        # Accept list of lists or np.ndarray
        m = Matrix.Identity(4)
        for i in range(4):
            for j in range(4):
                m[i][j] = float(transform_4x4[i][j])
        transform_4x4 = m
    mesh_obj.matrix_world = transform_4x4 @ mesh_obj.matrix_world


def cleanup_alignment_state(test_mesh, ref_mesh, context, reset_shading=True):
    """Post-ICP cleanup: remove PaintSelection color attr, restore Distance
    color attr if present, and reset viewport shading to MATERIAL mode.

    Mirrors the behavior previously duplicated across all ICP operators.
    """
    for mesh_obj in [test_mesh, ref_mesh]:
        if mesh_obj is None:
            continue
        mesh_data = mesh_obj.data
        if "PaintSelection" in mesh_data.color_attributes:
            mesh_data.color_attributes.remove(
                mesh_data.color_attributes["PaintSelection"]
            )
        if "Distance" in mesh_data.color_attributes:
            mesh_data.color_attributes.active_color = mesh_data.color_attributes["Distance"]
            mesh_data.color_attributes.render_color_index = (
                mesh_data.color_attributes.active_color_index
            )
        elif len(mesh_data.color_attributes) > 0:
            mesh_data.color_attributes.active_color = mesh_data.color_attributes[0]
            mesh_data.color_attributes.render_color_index = 0

    if reset_shading and context is not None:
        # Restore previously saved shading state if available, otherwise leave shading alone
        scn = context.scene
        saved_type = scn.get('_ma_saved_shading_type', None)
        saved_color = scn.get('_ma_saved_color_type', None)
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        if saved_type is not None:
                            try: space.shading.type = saved_type
                            except: pass
                        if saved_color is not None:
                            try: space.shading.color_type = saved_color
                            except: pass
        # Clear saved state
        if '_ma_saved_shading_type' in scn:
            del scn['_ma_saved_shading_type']
        if '_ma_saved_color_type' in scn:
            del scn['_ma_saved_color_type']


# ===================== PROPERTY GROUP =====================

class MeshAlignmentSettings(PropertyGroup):
    # Landmark radius for region-based initial alignment
    landmark_radius: FloatProperty(
        name="Landmark Radius (mm)",
        description="Radius around each landmark click - all vertices within this distance are used for alignment",
        default=5.0,
        min=0.5,
        max=20.0,
        subtype='DISTANCE'
    )
    
    # Mesh Selection
    test_mesh: PointerProperty(
        name="Test Mesh",
        type=bpy.types.Object,
        poll=lambda self, obj: obj and obj.type == 'MESH'
    )
    
    reference_mesh: PointerProperty(
        name="Reference Mesh", 
        type=bpy.types.Object,
        poll=lambda self, obj: obj and obj.type == 'MESH'
    )
    
    # Alignment Method Selection
    use_vertex_selection: BoolProperty(
        name="Use Vertex Paint Selection",
        description="Paint regions on meshes to select alignment areas",
        default=True
    )
    
    # Downsampling properties
    use_downsampling: BoolProperty(
        name="Use Downsampling",
        description="Enable downsampling to reduce computation time",
        default=False
    )
    
    downsampling_percentage: FloatProperty(
        name="Downsampling %",
        description="Percentage of vertices to remove (0-99%)",
        default=50.0,
        min=0.0,
        max=99.0,
        subtype='PERCENTAGE'
    )
    
    # Basic ICP parameters
    icp_iterations: IntProperty(
        name="ICP Iterations",
        description="Number of ICP iterations",
        default=50,
        min=1,
        max=200
    )
    
    outlier_percentage: FloatProperty(
        name="Outlier %",
        description="Percentage of outliers to reject",
        default=20.0,
        min=0.0,
        max=50.0,
        subtype='PERCENTAGE'
    )
    
    # Professional ICP Properties (Open3D)
    icp_algorithm: EnumProperty(
        name="ICP Algorithm",
        items=[
            ('POINT_TO_POINT', "Point-to-Point", "Classic ICP algorithm"),
            ('POINT_TO_PLANE', "Point-to-Plane", "More accurate but slower"),
            ('COLOR', "Colored ICP", "Uses color information if available"),
        ],
        default='POINT_TO_PLANE'
    )
    
    voxel_size: FloatProperty(
        name="Voxel Size",
        description="Voxel size for downsampling (0 = no downsampling)",
        default=0.005,
        min=0.0,
        max=0.1,
        precision=4,
        unit='LENGTH'
    )
    
    max_correspondence: FloatProperty(
        name="Max Correspondence Distance",
        description="Maximum correspondence distance for ICP",
        default=0.05,
        min=0.001,
        max=1.0,
        precision=4,
        unit='LENGTH'
    )
    
    max_iterations: IntProperty(
        name="Max Iterations",
        description="Maximum number of ICP iterations",
        default=50,
        min=1,
        max=200
    )
    
    create_aligned_copy: BoolProperty(
        name="Create Aligned Copy",
        description="Create a copy of the aligned mesh instead of moving original",
        default=True
    )
    
    # Status flags
    basic_alignment_complete: BoolProperty(default=False)
    professional_alignment_complete: BoolProperty(default=False)
    
    # Open3D availability
    open3d_available: BoolProperty(default=False)
    open3d_version: StringProperty(default="")
    
    # Dependency installation status
    dependency_installing: BoolProperty(default=False)

# ===================== HELPER FUNCTIONS =====================

def drawTextCallback(context, dummy):
    """Callback function for plotting landmark positions"""
    for object in bpy.context.visible_objects:
        if object.get('landmarkDictionary') is not None:
            for landmark, index in object['landmarkDictionary'].items():
                vertLoc = object.matrix_world @ object.data.vertices[index].co
                vertLocOnScreen = view3d_utils.location_3d_to_region_2d(context.region, context.space_data.region_3d, vertLoc)
                if vertLocOnScreen:
                    blf.position(0, vertLocOnScreen[0] - 2, vertLocOnScreen[1] - 8, 0)
                    blf.size(0, 20)
                    blf.color(0, 1, 1, 0, 1)
                    blf.draw(0, '·' + landmark)

def placeSeed(context, event):
    selectedObjects = bpy.context.selected_objects
    
    scene = context.scene
    region = context.region
    rv3d = context.region_data
    mouseCoordinates = event.mouse_region_x, event.mouse_region_y
    
    viewVector = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouseCoordinates)
    rayOrigin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouseCoordinates)
    rayTarget = rayOrigin + viewVector
    
    successArray = []
    hitLocationArray = []
    distanceArray = []
    
    for object in selectedObjects:
        matrixInverted = object.matrix_world.inverted()
        rayOriginObject = matrixInverted @ rayOrigin
        rayTargetObject = matrixInverted @ rayTarget
        rayVectorObject = rayTargetObject - rayOriginObject
        
        success, hitLocation, _, _ = object.ray_cast(rayOriginObject, rayVectorObject)
        
        successArray.append(success)
        hitLocationArray.append(hitLocation)
        distanceArray.append(np.linalg.norm(hitLocation - rayOriginObject))
        
    if np.all(successArray):
        object = selectedObjects[np.argmin(distanceArray)]
        hitLocation = hitLocationArray[np.argmin(distanceArray)]
    elif not np.any(successArray):
        return None, None
    else:
        object = selectedObjects[np.squeeze(np.where(successArray))]
        hitLocation = hitLocationArray[np.squeeze(np.where(successArray))]
    
    tree = mu.kdtree.KDTree(len(object.data.vertices))
    for i, v in enumerate(object.data.vertices):
        tree.insert(v.co, i)
    tree.balance()
    
    _, seedIndex, _ = tree.find(hitLocation)
    
    return object, seedIndex

# ===================== DEPENDENCY OPERATORS =====================

class MESH_OT_check_dependencies(Operator):
    """Check if Open3D is installed"""
    bl_idname = "mesh.check_dependencies"
    bl_label = "Check Dependencies"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        settings = context.scene.mesh_alignment
        is_installed, version = check_open3d_installed()
        
        settings.open3d_available = is_installed
        if is_installed:
            settings.open3d_version = version
            self.report({'INFO'}, f"Open3D {version} is installed and ready")
        else:
            settings.open3d_version = ""
            self.report({'WARNING'}, "Open3D is not installed")
        
        return {'FINISHED'}

class MESH_OT_install_open3d(Operator):
    """Install Open3D library and its dependencies"""
    bl_idname = "mesh.install_open3d"
    bl_label = "Install Open3D"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        settings = context.scene.mesh_alignment
        settings.dependency_installing = True
        
        try:
            # Get Python executable
            python_exe = get_python_executable()
            
            self.report({'INFO'}, "Installing Open3D and dependencies... This may take a few minutes.")
            
            # First install required dependencies that might be missing
            dependencies = ["markupsafe", "jinja2", "numpy"]
            
            for dep in dependencies:
                result = subprocess.run(
                    [python_exe, "-m", "pip", "install", dep],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"Warning: Could not install {dep}: {result.stderr}")
            
            # Now install/upgrade Open3D
            result = subprocess.run(
                [python_exe, "-m", "pip", "install", "--upgrade", "open3d"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Try to import to verify installation
                try:
                    import importlib
                    import open3d
                    importlib.reload(open3d)
                    
                    settings.open3d_available = True
                    settings.open3d_version = open3d.__version__
                    self.report({'INFO'}, f"Open3D {open3d.__version__} installed successfully!")
                except ImportError as e:
                    # May need Blender restart
                    self.report({'WARNING'}, f"Installation completed. Please restart Blender to use Open3D. ({str(e)})")
            else:
                self.report({'ERROR'}, f"Installation failed: {result.stderr}")
                
        except Exception as e:
            self.report({'ERROR'}, f"Installation error: {str(e)}")
        
        settings.dependency_installing = False
        return {'FINISHED'}

# ===================== ALIGNMENT OPERATORS =====================

class MESH_OT_select_vertices(Operator):
    """Enter edit mode to select vertices with circle select (C)"""
    bl_idname = "mesh.select_vertices_align"
    bl_label = "Select Vertices"
    bl_options = {'REGISTER', 'UNDO'}
    
    mesh_type: bpy.props.StringProperty()
    
    def execute(self, context):
        settings = context.scene.mesh_alignment
        
        if self.mesh_type == 'test':
            target_mesh = settings.test_mesh
            other_mesh = settings.reference_mesh
        else:
            target_mesh = settings.reference_mesh
            other_mesh = settings.test_mesh
            
        if not target_mesh:
            self.report({'ERROR'}, f"No {self.mesh_type} mesh selected")
            return {'CANCELLED'}
        
        # Exit any edit mode first
        if context.mode == 'EDIT_MESH':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        # Deselect all objects
        for obj in context.selected_objects:
            obj.select_set(False)
        
        # Show other mesh as wireframe for reference
        if other_mesh:
            other_mesh.display_type = 'WIRE'
            other_mesh.show_in_front = True
        
        # Enable X-ray for better visibility
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.show_xray = True
        
        # Select and enter edit mode on target mesh
        target_mesh.select_set(True)
        context.view_layer.objects.active = target_mesh
        bpy.ops.object.mode_set(mode='EDIT')
        
        other_name = "Reference" if self.mesh_type == 'test' else "Test"
        self.report({'INFO'}, f"Select vertices with C (circle select). {other_name} mesh shown as wireframe. Click Apply when done.")
        return {'FINISHED'}

class MESH_OT_paint_test_mesh(Operator):
    """Enter vertex paint mode on Test mesh"""
    bl_idname = "mesh.paint_test_mesh"
    bl_label = "Paint Test Mesh"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        settings = context.scene.mesh_alignment
        return settings.test_mesh is not None
    
    def execute(self, context):
        settings = context.scene.mesh_alignment
        mesh_obj = settings.test_mesh
        other_mesh = settings.reference_mesh
        
        # Exit any mode first
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        # Deselect all objects
        for obj in context.selected_objects:
            obj.select_set(False)
        
        # Ensure vertex color layer exists
        mesh = mesh_obj.data
        color_layer_name = "PaintSelection"
        
        if color_layer_name not in mesh.color_attributes:
            mesh.color_attributes.new(name=color_layer_name, type='FLOAT_COLOR', domain='POINT')
        
        color_layer = mesh.color_attributes[color_layer_name]
        mesh.color_attributes.active_color = color_layer
        mesh.color_attributes.render_color_index = mesh.color_attributes.active_color_index
        
        # Set all to white (unpainted) only if first time
        has_painted = False
        for i in range(len(color_layer.data)):
            r, g, b, a = color_layer.data[i].color
            if r < 0.95 or g < 0.95 or b < 0.95:
                has_painted = True
                break
        
        if not has_painted:
            for i in range(len(color_layer.data)):
                color_layer.data[i].color = (1.0, 1.0, 1.0, 1.0)
        
        # Make sure other mesh is visible and has PaintSelection for display
        if other_mesh:
            other_mesh.hide_set(False)
            other_data = other_mesh.data
            if color_layer_name not in other_data.color_attributes:
                other_data.color_attributes.new(name=color_layer_name, type='FLOAT_COLOR', domain='POINT')
                # Initialize to white
                other_layer = other_data.color_attributes[color_layer_name]
                for i in range(len(other_layer.data)):
                    other_layer.data[i].color = (1.0, 1.0, 1.0, 1.0)
            # Set PaintSelection as active on other mesh too for display
            other_data.color_attributes.active_color = other_data.color_attributes[color_layer_name]
        
        # Select and activate mesh
        mesh_obj.hide_set(False)
        mesh_obj.select_set(True)
        context.view_layer.objects.active = mesh_obj
        
        # Enter vertex paint mode
        bpy.ops.object.mode_set(mode='VERTEX_PAINT')
        
        # Set vertex paint brush color to PINK (multiple fallback methods)
        PAINT_COLOR = (1.0, 0.2, 0.4)
        try:
            # Method 1: Tool settings brush
            if context.tool_settings.vertex_paint and context.tool_settings.vertex_paint.brush:
                context.tool_settings.vertex_paint.brush.color = PAINT_COLOR

            # Method 2: Unified paint settings (Blender 5.0+ location - per paint mode)
            # In Blender 4.3+ / 5.0, unified_paint_settings moved under each paint mode.
            # If use_unified_color is True, the unified color overrides brush.color.
            try:
                ups = context.tool_settings.vertex_paint.unified_paint_settings
                ups.color = PAINT_COLOR
                ups.secondary_color = (1.0, 1.0, 1.0)
            except AttributeError:
                # Older Blender: unified_paint_settings was at tool_settings level
                try:
                    ups = context.tool_settings.unified_paint_settings
                    ups.color = PAINT_COLOR
                    ups.secondary_color = (1.0, 1.0, 1.0)
                except AttributeError:
                    pass

            # Method 3: Set all vertex paint brushes to pink
            for brush in bpy.data.brushes:
                if brush.use_paint_vertex:
                    brush.color = PAINT_COLOR
        except Exception as e:
            print(f"Brush color setting warning: {e}")
        
        # Save current shading state (so cleanup can restore Texture/Material Preview later)
        scn = context.scene
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        if '_ma_saved_shading_type' not in scn:
                            scn['_ma_saved_shading_type'] = space.shading.type
                            scn['_ma_saved_color_type'] = space.shading.color_type
                        space.shading.type = 'SOLID'
                        space.shading.color_type = 'VERTEX'
        
        self.report({'INFO'}, "Paint on Test mesh. F to change brush size. Tab when done.")
        return {'FINISHED'}


class MESH_OT_paint_reference_mesh(Operator):
    """Enter vertex paint mode on Reference mesh"""
    bl_idname = "mesh.paint_reference_mesh"
    bl_label = "Paint Reference Mesh"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        settings = context.scene.mesh_alignment
        return settings.reference_mesh is not None
    
    def execute(self, context):
        settings = context.scene.mesh_alignment
        mesh_obj = settings.reference_mesh
        other_mesh = settings.test_mesh
        
        # Exit any mode first
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        # Deselect all objects
        for obj in context.selected_objects:
            obj.select_set(False)
        
        # Ensure vertex color layer exists
        mesh = mesh_obj.data
        color_layer_name = "PaintSelection"
        
        if color_layer_name not in mesh.color_attributes:
            mesh.color_attributes.new(name=color_layer_name, type='FLOAT_COLOR', domain='POINT')
        
        color_layer = mesh.color_attributes[color_layer_name]
        mesh.color_attributes.active_color = color_layer
        mesh.color_attributes.render_color_index = mesh.color_attributes.active_color_index
        
        # Set all to white (unpainted) only if first time
        has_painted = False
        for i in range(len(color_layer.data)):
            r, g, b, a = color_layer.data[i].color
            if r < 0.95 or g < 0.95 or b < 0.95:
                has_painted = True
                break
        
        if not has_painted:
            for i in range(len(color_layer.data)):
                color_layer.data[i].color = (1.0, 1.0, 1.0, 1.0)
        
        # Make sure other mesh is visible and has PaintSelection for display
        if other_mesh:
            other_mesh.hide_set(False)
            other_data = other_mesh.data
            if color_layer_name not in other_data.color_attributes:
                other_data.color_attributes.new(name=color_layer_name, type='FLOAT_COLOR', domain='POINT')
                # Initialize to white
                other_layer = other_data.color_attributes[color_layer_name]
                for i in range(len(other_layer.data)):
                    other_layer.data[i].color = (1.0, 1.0, 1.0, 1.0)
            # Set PaintSelection as active on other mesh too for display
            other_data.color_attributes.active_color = other_data.color_attributes[color_layer_name]
        
        # Select and activate mesh
        mesh_obj.hide_set(False)
        mesh_obj.select_set(True)
        context.view_layer.objects.active = mesh_obj
        
        # Enter vertex paint mode
        bpy.ops.object.mode_set(mode='VERTEX_PAINT')
        
        # Set vertex paint brush color to PINK (multiple fallback methods)
        PAINT_COLOR = (1.0, 0.2, 0.4)
        try:
            # Method 1: Tool settings brush
            if context.tool_settings.vertex_paint and context.tool_settings.vertex_paint.brush:
                context.tool_settings.vertex_paint.brush.color = PAINT_COLOR

            # Method 2: Unified paint settings (Blender 5.0+ location - per paint mode)
            # In Blender 4.3+ / 5.0, unified_paint_settings moved under each paint mode.
            # If use_unified_color is True, the unified color overrides brush.color.
            try:
                ups = context.tool_settings.vertex_paint.unified_paint_settings
                ups.color = PAINT_COLOR
                ups.secondary_color = (1.0, 1.0, 1.0)
            except AttributeError:
                # Older Blender: unified_paint_settings was at tool_settings level
                try:
                    ups = context.tool_settings.unified_paint_settings
                    ups.color = PAINT_COLOR
                    ups.secondary_color = (1.0, 1.0, 1.0)
                except AttributeError:
                    pass

            # Method 3: Set all vertex paint brushes to pink
            for brush in bpy.data.brushes:
                if brush.use_paint_vertex:
                    brush.color = PAINT_COLOR
        except Exception as e:
            print(f"Brush color setting warning: {e}")
        
        # Save current shading state (so cleanup can restore Texture/Material Preview later)
        scn = context.scene
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        if '_ma_saved_shading_type' not in scn:
                            scn['_ma_saved_shading_type'] = space.shading.type
                            scn['_ma_saved_color_type'] = space.shading.color_type
                        space.shading.type = 'SOLID'
                        space.shading.color_type = 'VERTEX'
        
        self.report({'INFO'}, "Paint on Reference mesh. F to change brush size. Tab when done.")
        return {'FINISHED'}


class MESH_OT_accept_selection(Operator):
    """Accept painted vertices - converts paint to vertex selection"""
    bl_idname = "mesh.accept_selection"
    bl_label = "Accept Selection"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        settings = context.scene.mesh_alignment
        return settings.test_mesh and settings.reference_mesh
    
    def get_painted_indices(self, mesh_obj):
        """Get indices of vertices that are not white (painted with any color)"""
        mesh = mesh_obj.data
        painted_indices = []
        
        color_layer_name = "PaintSelection"
        if color_layer_name not in mesh.color_attributes:
            return painted_indices
        
        color_layer = mesh.color_attributes[color_layer_name]
        
        for i, color_data in enumerate(color_layer.data):
            r, g, b, a = color_data.color
            # If not white (any channel < 0.95), it's painted
            if r < 0.95 or g < 0.95 or b < 0.95:
                painted_indices.append(i)
        
        return painted_indices
    
    def execute(self, context):
        settings = context.scene.mesh_alignment
        
        # Exit vertex paint mode if in it
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        test_mesh = settings.test_mesh
        ref_mesh = settings.reference_mesh
        
        results = []
        
        # Process both meshes - convert paint to vertex selection
        for mesh_obj, mesh_type in [(test_mesh, 'Test'), (ref_mesh, 'Ref')]:
            painted_indices = self.get_painted_indices(mesh_obj)
            
            if len(painted_indices) < 3:
                results.append(f"{mesh_type}: 0")
                continue
            
            # Deselect all vertices first
            for v in mesh_obj.data.vertices:
                v.select = False
            
            # Select painted vertices
            for idx in painted_indices:
                mesh_obj.data.vertices[idx].select = True
            
            results.append(f"{mesh_type}: {len(painted_indices)}")
            
            # DON'T remove PaintSelection - let it stay for visual reference
            # Distance Analysis will create its own color attribute
        
        # Restore previously saved shading state (don't force MATERIAL - that breaks
        # DICOM slice textures). If no state was saved, leave shading untouched.
        scn = context.scene
        saved_type = scn.get('_ma_saved_shading_type', None)
        saved_color = scn.get('_ma_saved_color_type', None)
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        if saved_type is not None:
                            try: space.shading.type = saved_type
                            except: pass
                        if saved_color is not None:
                            try: space.shading.color_type = saved_color
                            except: pass
        if '_ma_saved_shading_type' in scn:
            del scn['_ma_saved_shading_type']
        if '_ma_saved_color_type' in scn:
            del scn['_ma_saved_color_type']
        
        # Select both meshes
        for obj in context.selected_objects:
            obj.select_set(False)
        test_mesh.select_set(True)
        ref_mesh.select_set(True)
        context.view_layer.objects.active = ref_mesh
        
        self.report({'INFO'}, f"Vertices selected: {', '.join(results)}")
        return {'FINISHED'}


class MESH_OT_place_landmarks(Operator):
    """Place landmarks for initial alignment"""
    bl_idname = "mesh.place_landmarks_align"
    bl_label = "Place Landmarks"
    bl_options = {'REGISTER', 'UNDO'}
    
    current_mesh_index: bpy.props.IntProperty(default=0)  # 0 = test mesh, 1 = reference mesh
    
    @classmethod
    def poll(cls, context):
        settings = context.scene.mesh_alignment
        return settings.test_mesh and settings.reference_mesh
    
    def modal(self, context, event):
        settings = context.scene.mesh_alignment
        
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            return {'PASS_THROUGH'}
            
        elif event.type in {'RET', 'NUMPAD_ENTER'}:
            return {'FINISHED'}
            
        elif event.type == 'ESC':
            # Clean up landmarks on cancel
            for obj in [settings.test_mesh, settings.reference_mesh]:
                if obj and obj.get('landmarkDictionary') is not None:
                    del obj['landmarkDictionary']
            context.area.tag_redraw()
            return {'CANCELLED'}
            
        elif event.type == 'RIGHTMOUSE' and event.value == 'PRESS':
            # Switch to next mesh
            if self.current_mesh_index == 0:
                # Switch from test to reference
                self.current_mesh_index = 1
                current_mesh = settings.reference_mesh
                
                # Deselect all and select reference mesh
                for obj in context.selected_objects:
                    obj.select_set(False)
                current_mesh.select_set(True)
                context.view_layer.objects.active = current_mesh
                
                self.report({'INFO'}, f"Switched to Reference mesh. Continue placing landmarks...")
                
            else:
                # Switch from reference - select both meshes and finish
                for obj in context.selected_objects:
                    obj.select_set(False)
                    
                settings.test_mesh.select_set(True)
                settings.reference_mesh.select_set(True)
                context.view_layer.objects.active = settings.reference_mesh
                
                self.report({'INFO'}, "Both meshes selected. Ready for alignment!")
                return {'FINISHED'}
                
            return {'RUNNING_MODAL'}
            
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            # Add draw handler if not already added
            bpy.types.SpaceView3D.draw_handler_add(drawTextCallback, (context, None), 'WINDOW', 'POST_PIXEL')
            
            # Get current target mesh
            if self.current_mesh_index == 0:
                target_mesh = settings.test_mesh
                mesh_name = "Test"
            else:
                target_mesh = settings.reference_mesh  
                mesh_name = "Reference"
            
            # Place landmark on the active mesh only
            result = self.placeSeedOnSpecificMesh(context, event, target_mesh)
            object, seedIndex, nearbyIndices = result if result is not None else (None, None, None)
            
            if object is not None and object == target_mesh:
                # Landmarks are stored in landmarkDictionary only; no region data
                if object.get('landmarkDictionary') is None:
                    object['landmarkDictionary'] = {}
                
                if seedIndex is None:
                    self.report({'ERROR'}, f"Cannot place landmark on {mesh_name} mesh.")
                else:
                    if seedIndex not in object['landmarkDictionary'].values():
                        landmark = str(len(object['landmarkDictionary']) + 1)
                        object['landmarkDictionary'].update({landmark: seedIndex})
                        landmark_count = len(object['landmarkDictionary'])
                        self.report({'INFO'}, f"Landmark {landmark} placed on {mesh_name} mesh ({landmark_count} total)")
                    else:
                        self.report({'ERROR'}, "Another landmark is already on this position.")
            else:
                self.report({'WARNING'}, f"Click on the {mesh_name} mesh to place landmarks")
            
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
            
        return {'RUNNING_MODAL'}
    
    def placeSeedOnSpecificMesh(self, context, event, target_mesh):
        """Place seed: returns center vertex index AND list of indices within landmark_radius."""
        scene = context.scene
        settings = scene.mesh_alignment
        region = context.region
        rv3d = context.region_data
        mouseCoordinates = event.mouse_region_x, event.mouse_region_y
        
        viewVector = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouseCoordinates)
        rayOrigin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouseCoordinates)
        rayTarget = rayOrigin + viewVector
        
        # Convert to object space
        matrixInverted = target_mesh.matrix_world.inverted()
        rayOriginObject = matrixInverted @ rayOrigin
        rayTargetObject = matrixInverted @ rayTarget
        rayVectorObject = rayTargetObject - rayOriginObject
        
        # Raycast on specific mesh
        success, hitLocation, _, _ = target_mesh.ray_cast(rayOriginObject, rayVectorObject)
        
        if not success:
            return None, None, None
        
        # Build kd tree
        tree = mu.kdtree.KDTree(len(target_mesh.data.vertices))
        for i, v in enumerate(target_mesh.data.vertices):
            tree.insert(v.co, i)
        tree.balance()
        
        # Center vertex (closest to hit)
        _, centerIndex, _ = tree.find(hitLocation)
        
        # All vertices within radius (object-space distance == world distance if no scale)
        radius = settings.landmark_radius
        # Account for object scale: convert radius to local space
        scale = target_mesh.matrix_world.to_scale()
        avg_scale = (scale.x + scale.y + scale.z) / 3.0
        local_radius = radius / max(avg_scale, 1e-6)
        
        nearby = tree.find_range(hitLocation, local_radius)
        nearbyIndices = [idx for (_co, idx, _dist) in nearby]
        
        if centerIndex not in nearbyIndices:
            nearbyIndices.append(centerIndex)
        
        return target_mesh, centerIndex, nearbyIndices
    
    def invoke(self, context, event):
        settings = context.scene.mesh_alignment
        
        # Start with test mesh
        self.current_mesh_index = 0
        
        # Select only test mesh initially
        for obj in context.selected_objects:
            obj.select_set(False)
            
        settings.test_mesh.select_set(True)
        context.view_layer.objects.active = settings.test_mesh
        
        self.report({'INFO'}, "Started with Test mesh. Left-click to place landmarks, Right-click to switch to Reference mesh.")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

class MESH_OT_delete_landmarks(Operator):
    """Delete landmarks of selected objects"""
    bl_idname = "mesh.delete_landmarks_align"
    bl_label = "Delete Landmarks"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        settings = context.scene.mesh_alignment
        return settings.test_mesh or settings.reference_mesh
            
    def execute(self, context):
        settings = context.scene.mesh_alignment
        deleted_count = 0
        
        for obj in [settings.test_mesh, settings.reference_mesh]:
            if obj and obj.get('landmarkDictionary') is not None:
                landmark_count = len(obj['landmarkDictionary'])
                del obj['landmarkDictionary']
                if obj.get('landmarkRegions') is not None:
                    try: del obj['landmarkRegions']
                    except: pass
                deleted_count += landmark_count
                
        if deleted_count > 0:
            self.report({'INFO'}, f"Deleted {deleted_count} landmarks")
        else:
            self.report({'WARNING'}, "No landmarks to delete")
            
        context.area.tag_redraw()
        return {'FINISHED'}

class MESH_OT_pca_align(Operator):
    """Auto Pre-Align (PCA): Coarse alignment using principal component analysis.
    
    Aligns the test mesh's principal axes to the reference mesh's principal axes.
    Tests all 8 axis-flip combinations (since PCA eigenvectors are sign-ambiguous)
    and picks the one with lowest Chamfer distance. Best run BEFORE landmark
    alignment or ICP when meshes are in arbitrary orientations."""
    bl_idname = "mesh.pca_align"
    bl_label = "Auto Pre-Align (PCA)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        s = context.scene.mesh_alignment
        return s.test_mesh and s.reference_mesh

    @staticmethod
    def _pca_frame(points):
        """Return (centroid, axes) where axes is 3x3 with rows = eigenvectors
        sorted by descending eigenvalue."""
        c = points.mean(axis=0)
        centered = points - c
        cov = (centered.T @ centered) / max(1, len(centered) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
        order = np.argsort(eigvals)[::-1]
        axes = eigvecs[:, order].T  # rows = principal axes, descending
        return c, axes

    @staticmethod
    def _chamfer(src, tgt_tree):
        """One-sided Chamfer distance: mean nearest-neighbor distance from src to tgt."""
        d, _ = tgt_tree.query(src, k=1, workers=-1)
        return float(d.mean())

    def execute(self, context):
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            self.report({'ERROR'}, "scipy not available - cannot run PCA alignment")
            return {'CANCELLED'}

        import time
        s = context.scene.mesh_alignment
        test_mesh = s.test_mesh
        ref_mesh = s.reference_mesh

        t_start = time.perf_counter()

        # Collect world-space vertices
        test_verts, _ = collect_world_vertices(test_mesh, prefer_selected=False)
        ref_verts, _ = collect_world_vertices(ref_mesh, prefer_selected=False)

        if len(test_verts) < 10 or len(ref_verts) < 10:
            self.report({'ERROR'}, "Both meshes need at least 10 vertices")
            return {'CANCELLED'}

        # Downsample for chamfer eval (we keep the FULL test verts only for centroid math)
        max_eval = 5000
        if len(test_verts) > max_eval:
            idx = np.random.RandomState(42).choice(len(test_verts), max_eval, replace=False)
            test_eval = test_verts[idx]
        else:
            test_eval = test_verts
        if len(ref_verts) > max_eval:
            idx = np.random.RandomState(42).choice(len(ref_verts), max_eval, replace=False)
            ref_eval = ref_verts[idx]
        else:
            ref_eval = ref_verts

        # Compute PCA frames on the eval point clouds
        test_c, test_axes = self._pca_frame(test_eval)
        ref_c, ref_axes = self._pca_frame(ref_eval)

        # Force both PCA frames to be right-handed (det = +1)
        # by flipping the smallest-eigenvalue axis if necessary.
        if np.linalg.det(test_axes) < 0:
            test_axes[2] *= -1
        if np.linalg.det(ref_axes) < 0:
            ref_axes[2] *= -1

        # Build KDTree on reference for chamfer eval
        ref_tree = cKDTree(ref_eval)

        # Test the 4 valid axis-sign combinations (sign flips that preserve
        # right-handedness: even number of minus signs).
        # Combinations: (+,+,+), (-,-,+), (-,+,-), (+,-,-)
        best_score = float('inf')
        best_R = None

        test_centered = test_eval - test_c

        valid_signs = [(1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)]
        for sx, sy, sz in valid_signs:
            flipped_test = test_axes * np.array([[sx], [sy], [sz]])
            # Rotation maps test PCA basis -> reference PCA basis
            R = ref_axes.T @ flipped_test
            # Sanity check (should always be ~+1 now)
            if np.linalg.det(R) < 0.5:
                continue

            transformed = test_centered @ R.T + ref_c
            score = self._chamfer(transformed, ref_tree)
            if score < best_score:
                best_score = score
                best_R = R

        if best_R is None:
            self.report({'ERROR'}, "PCA alignment failed - no valid rotation")
            return {'CANCELLED'}

        # Build the 4x4 transform: T_to_origin -> R -> T_to_ref_c
        T1 = Matrix.Translation(-Vector(test_c.tolist()))
        R4 = Matrix.Identity(4)
        for i in range(3):
            for j in range(3):
                R4[i][j] = float(best_R[i][j])
        T2 = Matrix.Translation(Vector(ref_c.tolist()))
        transform = T2 @ R4 @ T1

        # Apply via helper
        apply_world_transform(test_mesh, transform)

        elapsed = time.perf_counter() - t_start
        self.report({'INFO'},
            f"PCA align: chamfer {best_score:.3f}, {elapsed:.2f}s "
            f"({len(test_eval)}v vs {len(ref_eval)}v)")
        return {'FINISHED'}


class MESH_OT_initial_alignment(Operator):
    """Perform initial alignment using landmarks OR selected vertices"""
    bl_idname = "mesh.initial_alignment"
    bl_label = "Perform Initial Alignment"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        settings = context.scene.mesh_alignment
        return settings.test_mesh and settings.reference_mesh
    
    def execute(self, context):
        settings = context.scene.mesh_alignment
        
        fixedObject = settings.reference_mesh
        movingObject = settings.test_mesh
        
        # Smart mode detection: if landmarks exist, use them; otherwise use vertex selection
        has_landmarks = (fixedObject.get('landmarkDictionary') is not None and 
                        movingObject.get('landmarkDictionary') is not None and
                        len(fixedObject.get('landmarkDictionary', {})) >= 3 and
                        len(movingObject.get('landmarkDictionary', {})) >= 3)
        
        # Decide mode: landmarks take priority if they exist
        use_vertices = settings.use_vertex_selection and not has_landmarks
        
        if use_vertices:
            # Get selected vertices from both meshes (works with both paint and edit mode selection)
            movingArray = np.array([movingObject.matrix_world @ v.co for v in movingObject.data.vertices if v.select])
            fixedArray = np.array([fixedObject.matrix_world @ v.co for v in fixedObject.data.vertices if v.select])
            
            if len(movingArray) < 3:
                self.report({'ERROR'}, f"Test mesh: Need at least 3 selected vertices (has {len(movingArray)})")
                return {'CANCELLED'}
            
            if len(fixedArray) < 3:
                self.report({'ERROR'}, f"Reference mesh: Need at least 3 selected vertices (has {len(fixedArray)})")
                return {'CANCELLED'}
            
            # Calculate centroids
            fixedCentroid = np.mean(fixedArray, axis=0)
            movingCentroid = np.mean(movingArray, axis=0)
            
            # Move arrays to origin
            fixedOrigin = fixedArray - fixedCentroid
            movingOrigin = movingArray - movingCentroid
            
            # Build KD-tree for correspondence (like ICP)
            fixedTree = mu.kdtree.KDTree(len(fixedOrigin))
            for i, v in enumerate(fixedOrigin):
                fixedTree.insert(Vector(v), i)
            fixedTree.balance()
            
            # Find corresponding pairs
            movingPairs = []
            fixedPairs = []
            for mv in movingOrigin:
                _, idx, _ = fixedTree.find(Vector(mv))
                movingPairs.append(mv)
                fixedPairs.append(fixedOrigin[idx])
            
            movingPairs = np.array(movingPairs)
            fixedPairs = np.array(fixedPairs)
            
            # SVD for rotation (adapted from 3D OPERATORS ICP Registration)
            covMatrix = movingPairs.T @ fixedPairs
            U, s, Vt = np.linalg.svd(covMatrix)
            V = Vt.T
            rotation3x3 = V @ U.T
            
            # Prevent reflection
            if np.linalg.det(rotation3x3) < 0:
                V[:, -1] *= -1
                rotation3x3 = V @ U.T
            
            # Build rotation matrix
            rotationMatrix = np.eye(4)
            rotationMatrix[0:3, 0:3] = rotation3x3
            movingObject.matrix_world = mu.Matrix(rotationMatrix) @ movingObject.matrix_world
            
            # Translation (after rotation)
            translationMatrix = np.eye(4)
            translationMatrix[0:3, 3] = fixedCentroid - rotation3x3 @ movingCentroid
            movingObject.matrix_world = mu.Matrix(translationMatrix) @ movingObject.matrix_world
            
            # Clean up: Remove PaintSelection and restore color attributes
            cleanup_alignment_state(movingObject, fixedObject, context)
            
            self.report({'INFO'}, f"Initial alignment: {len(movingArray)} test, {len(fixedArray)} ref vertices")
            return {'FINISHED'}
            
        else:
            # Use landmark-based alignment with REGIONS (5mm radius around each click)
            if fixedObject.get('landmarkDictionary') is None or movingObject.get('landmarkDictionary') is None:
                self.report({'ERROR'}, "Both meshes must have landmarks (or use vertex selection mode)")
                return {'CANCELLED'}
                
            fixedLandmarks = fixedObject['landmarkDictionary']
            movingLandmarks = movingObject['landmarkDictionary']
            
            if len(fixedLandmarks) != len(movingLandmarks):
                self.report({'ERROR'}, f"Landmark count mismatch: Test has {len(movingLandmarks)}, Reference has {len(fixedLandmarks)}")
                return {'CANCELLED'}
                
            if len(fixedLandmarks) < 3:
                self.report({'ERROR'}, "At least 3 landmarks required")
                return {'CANCELLED'}
            
            # One point per landmark (the clicked vertex); no region averaging
            fixedPoints = []
            movingPoints = []
            for key in sorted(fixedLandmarks.keys()):
                if key not in movingLandmarks:
                    continue
                fidx = int(fixedLandmarks[key])
                midx = int(movingLandmarks[key])
                fixedPoints.append(fixedObject.matrix_world @ fixedObject.data.vertices[fidx].co)
                movingPoints.append(movingObject.matrix_world @ movingObject.data.vertices[midx].co)
            
            fixedPoints = np.array([list(p) for p in fixedPoints])
            movingPoints = np.array([list(p) for p in movingPoints])
            
            print(f"[Initial Alignment] {len(fixedPoints)} landmark pairs")
        
        # Calculate transformation using SVD (centroid-based alignment)
        fixedCentroid = np.mean(fixedPoints, axis=0)
        movingCentroid = np.mean(movingPoints, axis=0)
        
        fixedCentered = fixedPoints - fixedCentroid
        movingCentered = movingPoints - movingCentroid
        
        H = movingCentered.T @ fixedCentered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        t = fixedCentroid - R @ movingCentroid
        
        # Build transformation matrix
        transform = Matrix.Identity(4)
        for i in range(3):
            for j in range(3):
                transform[i][j] = R[i, j]
            transform[i][3] = t[i]
        
        # Apply transformation
        movingObject.matrix_world = transform @ movingObject.matrix_world
        
        # Delete landmarks after alignment (removes the numbers from viewport)
        for obj in (fixedObject, movingObject):
            for key in ('landmarkDictionary', 'landmarkRegions'):
                if obj.get(key) is not None:
                    try:
                        del obj[key]
                    except: pass
        
        self.report({'INFO'}, "Initial alignment completed")
        return {'FINISHED'}


class MESH_OT_local_icp(Operator):
    """Local ICP - Open3D Point-to-Plane ICP using ONLY selected/painted vertices"""
    bl_idname = "mesh.local_icp"
    bl_label = "Local ICP"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        settings = context.scene.mesh_alignment
        if not (settings.test_mesh and settings.reference_mesh):
            return False
        # Check if there are selected vertices
        test_selected = sum(1 for v in settings.test_mesh.data.vertices if v.select)
        ref_selected = sum(1 for v in settings.reference_mesh.data.vertices if v.select)
        return test_selected >= 4 and ref_selected >= 4
    
    def execute(self, context):
        # Check Open3D
        try:
            import open3d as o3d
        except ImportError:
            self.report({'ERROR'}, "Open3D not installed")
            return {'CANCELLED'}
        
        settings = context.scene.mesh_alignment
        
        test_mesh = settings.test_mesh
        ref_mesh = settings.reference_mesh
        
        # Get ONLY selected vertices
        test_verts = np.array([test_mesh.matrix_world @ v.co for v in test_mesh.data.vertices if v.select])
        ref_verts = np.array([ref_mesh.matrix_world @ v.co for v in ref_mesh.data.vertices if v.select])
        
        if len(test_verts) < 4:
            self.report({'ERROR'}, f"Test mesh: Need at least 4 selected vertices (has {len(test_verts)})")
            return {'CANCELLED'}
        
        if len(ref_verts) < 4:
            self.report({'ERROR'}, f"Reference mesh: Need at least 4 selected vertices (has {len(ref_verts)})")
            return {'CANCELLED'}
        
        original_test_count = len(test_verts)
        original_ref_count = len(ref_verts)
        
        self.report({'INFO'}, f"Running Local ICP (Open3D) on {original_test_count} vs {original_ref_count} selected vertices...")
        
        try:
            # Create Open3D point clouds
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(test_verts)
            
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(ref_verts)
            
            # Estimate normals for point-to-plane ICP
            source_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
            target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
            
            # Calculate threshold based on selected region size
            bbox = source_pcd.get_axis_aligned_bounding_box()
            max_extent = max(bbox.get_extent())
            threshold = max_extent * 0.05  # 5% of bounding box for local regions
            
            # Run Point-to-Plane ICP with Tukey robust kernel
            # TukeyLoss redescending: residuals beyond k get ZERO weight (full rejection),
            # not just down-weighted (Huber). Ideal when paint may include a few wrong vertices.
            # k = threshold * 0.5: residuals > 50% of threshold are treated as outliers.
            tukey_loss = o3d.pipelines.registration.TukeyLoss(k=threshold * 0.5)
            estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane(tukey_loss)

            result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd,
                threshold,
                np.eye(4),
                estimator,
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=settings.icp_iterations
                )
            )
            
            # Apply transformation
            transform_np = np.array(result.transformation)
            
            transform = Matrix.Identity(4)
            for i in range(4):
                for j in range(4):
                    transform[i][j] = transform_np[i, j]
            
            test_mesh.matrix_world = transform @ test_mesh.matrix_world
            
            fitness = result.fitness
            rmse = result.inlier_rmse
            
            # Clean up: Remove PaintSelection and restore color attributes
            cleanup_alignment_state(test_mesh, ref_mesh, context)
            
            self.report({'INFO'}, f"Local ICP completed. Fitness: {fitness:.2%}, RMSE: {rmse:.4f}")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Local ICP failed: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class MESH_OT_ai_icp(Operator):
    """GPU ICP - Fast GPU-accelerated ICP using PyTorch (runs in separate process)"""
    bl_idname = "mesh.ai_icp"
    bl_label = "GPU ICP"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        settings = context.scene.mesh_alignment
        # Only need test and reference mesh - selection is optional
        return settings.test_mesh and settings.reference_mesh
    
    def write_ply(self, filepath, vertices):
        """Write vertices to PLY file"""
        with open(filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
    
    def _try_fast_worker(self, context):
        """Try persistent worker path. Returns 'FINISHED' / None.
        None = fall through to legacy subprocess path.
        """
        import tempfile
        import time as _time
        import json as _json

        try:
            from gpu_icp_worker import GPUICPWorker
        except Exception as e:
            print(f"[GPU ICP] worker module not available, fallback: {e}")
            return None

        addon_dir = os.path.dirname(os.path.realpath(__file__))
        runner_v2 = os.path.join(addon_dir, "ai_icp_runner_v2.py")
        if not os.path.exists(runner_v2):
            print(f"[GPU ICP] runner_v2 not found, fallback")
            return None

        settings = context.scene.mesh_alignment
        test_mesh = settings.test_mesh
        ref_mesh = settings.reference_mesh

        # Collect vertices (selected or all)
        test_selected = [v for v in test_mesh.data.vertices if v.select]
        ref_selected = [v for v in ref_mesh.data.vertices if v.select]
        if len(test_selected) < 4:
            test_verts = np.array([test_mesh.matrix_world @ v.co for v in test_mesh.data.vertices], dtype=np.float32)
            use_full_test = True
        else:
            test_verts = np.array([test_mesh.matrix_world @ v.co for v in test_selected], dtype=np.float32)
            use_full_test = False
        if len(ref_selected) < 4:
            ref_verts = np.array([ref_mesh.matrix_world @ v.co for v in ref_mesh.data.vertices], dtype=np.float32)
            use_full_ref = True
        else:
            ref_verts = np.array([ref_mesh.matrix_world @ v.co for v in ref_selected], dtype=np.float32)
            use_full_ref = False

        original_test_count = len(test_verts)
        original_ref_count = len(ref_verts)

        # Write npy (binary, fast)
        temp_dir = tempfile.gettempdir()
        src_npy = os.path.join(temp_dir, "ai_icp_source.npy")
        tgt_npy = os.path.join(temp_dir, "ai_icp_target.npy")
        out_json = os.path.join(temp_dir, "ai_icp_result_v2.json")
        np.save(src_npy, test_verts)
        np.save(tgt_npy, ref_verts)

        worker = GPUICPWorker.get(runner_v2)
        if not worker.is_alive():
            self.report({'INFO'}, "GPU ICP: starting persistent worker (one-time setup)...")

        t0 = _time.perf_counter()
        ok, data = worker.run_icp(
            src_npy, tgt_npy, out_json,
            iters=settings.icp_iterations,
            outlier_pct=settings.outlier_percentage,
            timeout=300,
        )
        elapsed = _time.perf_counter() - t0

        if not ok:
            print(f"[GPU ICP] worker path failed: {data}, falling back to legacy")
            return None

        if not os.path.exists(out_json):
            print(f"[GPU ICP] worker reported done but no output, fallback")
            return None
        with open(out_json, 'r') as _f:
            icp_result = _json.load(_f)
        if not icp_result.get('success', False):
            return None

        transform_list = icp_result['transform']
        rmse = icp_result['rmse']
        transform = Matrix.Identity(4)
        for i in range(4):
            for j in range(4):
                transform[i][j] = transform_list[i][j]
        test_mesh.matrix_world = transform @ test_mesh.matrix_world

        for f in [src_npy, tgt_npy, out_json]:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass

        cleanup_alignment_state(test_mesh, ref_mesh, context)

        self.report({'INFO'}, f"GPU ICP (fast worker, {worker.device_name}): {original_test_count}v vs {original_ref_count}v, RMSE {rmse:.4f}, {elapsed:.2f}s")
        return {'FINISHED'}

    def execute(self, context):
        import subprocess
        import tempfile
        import json
        
        # Try fast persistent worker first; fall back to legacy on any failure
        fast_result = self._try_fast_worker(context)
        if fast_result is not None:
            return fast_result

        # ----- Legacy fallback path (original subprocess code below) -----
        settings = context.scene.mesh_alignment
        
        test_mesh = settings.test_mesh
        ref_mesh = settings.reference_mesh
        
        # Get selected vertices, or ALL vertices if none selected
        test_selected = [v for v in test_mesh.data.vertices if v.select]
        ref_selected = [v for v in ref_mesh.data.vertices if v.select]
        
        # If no selection, use all vertices
        if len(test_selected) < 4:
            test_verts = np.array([test_mesh.matrix_world @ v.co for v in test_mesh.data.vertices])
            use_full_test = True
        else:
            test_verts = np.array([test_mesh.matrix_world @ v.co for v in test_selected])
            use_full_test = False
        
        if len(ref_selected) < 4:
            ref_verts = np.array([ref_mesh.matrix_world @ v.co for v in ref_mesh.data.vertices])
            use_full_ref = True
        else:
            ref_verts = np.array([ref_mesh.matrix_world @ v.co for v in ref_selected])
            use_full_ref = False
        
        original_test_count = len(test_verts)
        original_ref_count = len(ref_verts)
        
        mode_str = "full mesh" if (use_full_test and use_full_ref) else "selected vertices"
        
        # Create temp files
        temp_dir = tempfile.gettempdir()
        source_ply = os.path.join(temp_dir, "ai_icp_source.ply")
        target_ply = os.path.join(temp_dir, "ai_icp_target.ply")
        output_json = os.path.join(temp_dir, "ai_icp_result.json")
        
        # Write PLY files
        self.write_ply(source_ply, test_verts)
        self.write_ply(target_ply, ref_verts)
        
        # Find the runner script
        addon_dir = os.path.dirname(os.path.realpath(__file__))
        runner_script = os.path.join(addon_dir, "ai_icp_runner.py")
        
        if not os.path.exists(runner_script):
            self.report({'ERROR'}, f"GPU ICP runner not found: {runner_script}")
            return {'CANCELLED'}
        
        # Run subprocess
        self.report({'INFO'}, f"Running GPU ICP ({mode_str}): {original_test_count} vs {original_ref_count} vertices...")
        
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    runner_script,
                    source_ply,
                    target_ply,
                    output_json,
                    str(settings.icp_iterations),
                    str(settings.outlier_percentage)
                ],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                self.report({'ERROR'}, f"GPU ICP failed: {result.stderr}")
                return {'CANCELLED'}
            
            # Read result
            if not os.path.exists(output_json):
                self.report({'ERROR'}, "GPU ICP did not produce output")
                return {'CANCELLED'}
            
            with open(output_json, 'r') as f:
                icp_result = json.load(f)
            
            if not icp_result.get('success', False):
                self.report({'ERROR'}, "GPU ICP failed")
                return {'CANCELLED'}
            
            # Apply transform
            transform_list = icp_result['transform']
            rmse = icp_result['rmse']
            
            transform = Matrix.Identity(4)
            for i in range(4):
                for j in range(4):
                    transform[i][j] = transform_list[i][j]
            
            test_mesh.matrix_world = transform @ test_mesh.matrix_world
            
            # Clean up temp files
            for f in [source_ply, target_ply, output_json]:
                if os.path.exists(f):
                    os.remove(f)
            
            # Clean up color attributes
            cleanup_alignment_state(test_mesh, ref_mesh, context)
            
            self.report({'INFO'}, f"GPU ICP completed. RMSE: {rmse:.4f}")
            return {'FINISHED'}
            
        except subprocess.TimeoutExpired:
            self.report({'ERROR'}, "GPU ICP timed out (>5 minutes)")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"GPU ICP failed: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class MESH_OT_open3d_icp(Operator):
    """Open3D ICP - Point-to-plane ICP using Open3D library"""
    bl_idname = "mesh.open3d_icp"
    bl_label = "Open3D ICP"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        settings = context.scene.mesh_alignment
        return settings.test_mesh and settings.reference_mesh
    
    def execute(self, context):
        # Check Open3D
        try:
            import open3d as o3d
        except ImportError:
            self.report({'ERROR'}, "Open3D not installed")
            return {'CANCELLED'}
        
        settings = context.scene.mesh_alignment
        
        test_mesh = settings.test_mesh
        ref_mesh = settings.reference_mesh
        
        # Get vertices (selected or all)
        test_selected = [v for v in test_mesh.data.vertices if v.select]
        ref_selected = [v for v in ref_mesh.data.vertices if v.select]
        
        if len(test_selected) < 4:
            test_verts = np.array([test_mesh.matrix_world @ v.co for v in test_mesh.data.vertices])
            use_full_test = True
        else:
            test_verts = np.array([test_mesh.matrix_world @ v.co for v in test_selected])
            use_full_test = False
        
        if len(ref_selected) < 4:
            ref_verts = np.array([ref_mesh.matrix_world @ v.co for v in ref_mesh.data.vertices])
            use_full_ref = True
        else:
            ref_verts = np.array([ref_mesh.matrix_world @ v.co for v in ref_selected])
            use_full_ref = False
        
        mode_str = "full mesh" if (use_full_test and use_full_ref) else "selected vertices"
        self.report({'INFO'}, f"Running Open3D ICP ({mode_str})...")
        
        try:
            # Create Open3D point clouds
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(test_verts)
            
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(ref_verts)
            
            # Estimate normals for point-to-plane ICP
            source_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
            target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
            
            # Calculate threshold based on mesh size
            bbox = source_pcd.get_axis_aligned_bounding_box()
            max_extent = max(bbox.get_extent())
            threshold = max_extent * 0.02  # 2% of bounding box
            
            # Run Point-to-Plane ICP
            result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd,
                threshold,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=settings.max_iterations
                )
            )
            
            # Apply transformation
            transform_np = np.array(result.transformation)
            
            transform = Matrix.Identity(4)
            for i in range(4):
                for j in range(4):
                    transform[i][j] = transform_np[i, j]
            
            test_mesh.matrix_world = transform @ test_mesh.matrix_world
            
            # Calculate fitness
            fitness = result.fitness
            rmse = result.inlier_rmse
            
            # Clean up color attributes
            cleanup_alignment_state(test_mesh, ref_mesh, context)
            
            self.report({'INFO'}, f"Open3D ICP completed. Fitness: {fitness:.2%}, RMSE: {rmse:.4f}")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Open3D ICP failed: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}



class MESH_OT_run_basic_icp(Operator):
    """Run basic ICP alignment - uses selected vertices if 'Use Vertex Selection' is enabled"""
    bl_idname = "mesh.run_basic_icp"
    bl_label = "Run Basic ICP"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        settings = context.scene.mesh_alignment
        return settings.test_mesh and settings.reference_mesh
    
    def execute(self, context):
        settings = context.scene.mesh_alignment
        
        test_mesh = settings.test_mesh
        ref_mesh = settings.reference_mesh
        
        # ICP does NOT use landmarks. Only:
        #  1) use_vertex_selection -> selected/painted vertices
        #  2) Else -> all vertices
        if settings.use_vertex_selection:
            test_verts = np.array([test_mesh.matrix_world @ v.co for v in test_mesh.data.vertices if v.select])
            ref_verts = np.array([ref_mesh.matrix_world @ v.co for v in ref_mesh.data.vertices if v.select])
            
            if len(test_verts) < 4:
                self.report({'ERROR'}, f"Test mesh: Need at least 4 selected vertices (has {len(test_verts)}). Either paint a region or disable 'Use Vertex Selection'.")
                return {'CANCELLED'}
            if len(ref_verts) < 4:
                self.report({'ERROR'}, f"Reference mesh: Need at least 4 selected vertices (has {len(ref_verts)}). Either paint a region or disable 'Use Vertex Selection'.")
                return {'CANCELLED'}
            print(f"[Basic ICP] Using selected vertices: {len(test_verts)} test / {len(ref_verts)} ref")
        else:
            test_verts = np.array([test_mesh.matrix_world @ v.co for v in test_mesh.data.vertices])
            ref_verts = np.array([ref_mesh.matrix_world @ v.co for v in ref_mesh.data.vertices])
            print(f"[Basic ICP] Using all vertices: {len(test_verts)} test / {len(ref_verts)} ref")
        
        if len(test_verts) < 4 or len(ref_verts) < 4:
            self.report({'ERROR'}, "Not enough vertices for ICP (minimum 4)")
            return {'CANCELLED'}
        
        # Store original vertex count for reporting
        original_test_count = len(test_verts)
        original_ref_count = len(ref_verts)
        
        # Apply downsampling if enabled
        if settings.use_downsampling:
            keep_ratio = (100 - settings.downsampling_percentage) / 100
            test_indices = np.random.choice(len(test_verts), max(4, int(len(test_verts) * keep_ratio)), replace=False)
            ref_indices = np.random.choice(len(ref_verts), max(4, int(len(ref_verts) * keep_ratio)), replace=False)
            test_verts = test_verts[test_indices]
            ref_verts = ref_verts[ref_indices]
        
        # Build KD-tree for reference
        tree = mu.kdtree.KDTree(len(ref_verts))
        for i, v in enumerate(ref_verts):
            tree.insert(Vector(v), i)
        tree.balance()
        
        # ICP iterations
        transform = Matrix.Identity(4)
        
        for iteration in range(settings.icp_iterations):
            # Find correspondences
            correspondences = []
            distances = []
            
            for test_v in test_verts:
                _, idx, dist = tree.find(Vector(test_v))
                correspondences.append(idx)
                distances.append(dist)
            
            # Remove outliers
            threshold_idx = int(len(distances) * (1 - settings.outlier_percentage / 100))
            threshold = sorted(distances)[threshold_idx]
            
            valid_test = []
            valid_ref = []
            
            for i, (test_v, corr_idx, dist) in enumerate(zip(test_verts, correspondences, distances)):
                if dist <= threshold:
                    valid_test.append(test_v)
                    valid_ref.append(ref_verts[corr_idx])
            
            valid_test = np.array(valid_test)
            valid_ref = np.array(valid_ref)
            
            if len(valid_test) < 4:
                break
            
            # Calculate transformation
            test_centroid = np.mean(valid_test, axis=0)
            ref_centroid = np.mean(valid_ref, axis=0)
            
            test_centered = valid_test - test_centroid
            ref_centered = valid_ref - ref_centroid
            
            H = test_centered.T @ ref_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            t = ref_centroid - R @ test_centroid
            
            # Build iteration transform
            iter_transform = Matrix.Identity(4)
            for i in range(3):
                for j in range(3):
                    iter_transform[i][j] = R[i, j]
                iter_transform[i][3] = t[i]
            
            # Apply to test vertices
            test_verts = np.array([iter_transform @ Vector(v) for v in test_verts])
            
            # Accumulate transform
            transform = iter_transform @ transform
        
        # Apply final transformation to mesh
        test_mesh.matrix_world = transform @ test_mesh.matrix_world
        
        # Also update painted mesh if it exists
        if settings.use_vertex_selection:
            test_painted = bpy.data.objects.get(test_mesh.name + "_painted")
            if test_painted:
                test_painted.matrix_world = transform @ test_painted.matrix_world
        
        settings.basic_alignment_complete = True
        
        # Calculate final RMSE
        final_distances = []
        for test_v in test_verts:
            _, _, dist = tree.find(Vector(test_v))
            final_distances.append(dist)
        rmse = np.sqrt(np.mean(np.array(final_distances) ** 2))
        
        # Clean up: Remove PaintSelection and restore color attributes
        cleanup_alignment_state(test_mesh, ref_mesh, context)
        
        if settings.use_vertex_selection:
            self.report({'INFO'}, f"ICP completed on {original_test_count} selected vertices. RMSE: {rmse:.4f}")
        else:
            self.report({'INFO'}, f"Basic ICP completed. RMSE: {rmse:.4f}")
        return {'FINISHED'}

# ===================== UTILITY OPERATORS =====================

class MESH_OT_fix_color_display(Operator):
    """Fix color attribute display for Distance Analysis"""
    bl_idname = "mesh.fix_color_display"
    bl_label = "Fix Color Display"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        fixed_count = 0
        for obj in context.scene.objects:
            if obj.type == 'MESH':
                mesh = obj.data
                if "Distance" in mesh.color_attributes:
                    mesh.color_attributes.active_color = mesh.color_attributes["Distance"]
                    mesh.color_attributes.render_color_index = mesh.color_attributes.active_color_index
                    fixed_count += 1
                elif len(mesh.color_attributes) > 0:
                    mesh.color_attributes.active_color = mesh.color_attributes[0]
                    mesh.color_attributes.render_color_index = 0
                    fixed_count += 1
        
        # Set viewport to vertex color
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.color_type = 'VERTEX'
        
        self.report({'INFO'}, f"Fixed color display for {fixed_count} mesh(es)")
        return {'FINISHED'}

# ===================== PANELS =====================

class VIEW3D_PT_mesh_alignment_main(Panel):
    """Main panel"""
    bl_label = "Mesh Alignment"
    bl_idname = "VIEW3D_PT_mesh_alignment_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Mesh Align"
    
    def draw_header(self, context):
        layout = self.layout
        # Yesil CHECKMARK ikonu
        layout.label(text="", icon='KEYTYPE_JITTER_VEC')
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.mesh_alignment
        
        # Mesh selection
        layout.prop(settings, "test_mesh", icon='MESH_DATA')
        layout.prop(settings, "reference_mesh", icon='MESH_DATA')
        
        if settings.test_mesh and settings.reference_mesh:
            # Show mesh info
            box = layout.box()
            box.label(text="Mesh Information:", icon='INFO')
            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text=f"Test: {len(settings.test_mesh.data.vertices):,} vertices")
            col.label(text=f"Reference: {len(settings.reference_mesh.data.vertices):,} vertices")
        
        # Utility section
        layout.separator()
        layout.operator("mesh.fix_color_display", icon='COLOR')

class VIEW3D_PT_basic_alignment(Panel):
    """Basic Alignment panel"""
    bl_label = "Basic Alignment"
    bl_idname = "VIEW3D_PT_basic_alignment"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Mesh Align"
    bl_parent_id = "VIEW3D_PT_mesh_alignment_main"
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.mesh_alignment
        
        # ==================== DEPENDENCY CHECK ====================
        box = layout.box()
        
        # Check Open3D status (read-only check, don't write to settings in draw)
        is_installed, version = check_open3d_installed()
        
        if is_installed:
            # Show success status
            row = box.row()
            row.label(text=f"Open3D {version} ✓", icon='CHECKMARK')
        else:
            # Show install button
            row = box.row()
            row.operator("mesh.check_dependencies", text="Check Dependencies", icon='FILE_REFRESH')
            
            if not settings.dependency_installing:
                row = box.row()
                row.alert = True
                row.operator("mesh.install_open3d", text="Install Open3D", icon='IMPORT')
            else:
                row = box.row()
                row.label(text="Installing...", icon='SORTTIME')
        
        layout.separator()
        
        # ==================== AUTO PRE-ALIGN (PCA) ====================
        box = layout.box()
        box.label(text="Auto Pre-Align", icon='SHADERFX')
        box.label(text="Coarse alignment via PCA - run BEFORE landmarks", icon='INFO')
        row = box.row()
        row.scale_y = 1.2
        row.operator("mesh.pca_align", text="Auto Pre-Align (PCA)", icon='ORIENTATION_GIMBAL')

        # ==================== LANDMARK ALIGNMENT ====================
        layout.separator()
        box = layout.box()
        box.label(text="Landmark Alignment", icon='PIVOT_CURSOR')
        
        # Show landmark count with current active mesh info
        if settings.test_mesh and settings.reference_mesh:
            test_landmarks = len(settings.test_mesh.get('landmarkDictionary', {}))
            ref_landmarks = len(settings.reference_mesh.get('landmarkDictionary', {}))
            
            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text=f"Test: {test_landmarks} landmarks")
            col.label(text=f"Reference: {ref_landmarks} landmarks")
            
            if test_landmarks > 0 and ref_landmarks > 0:
                if test_landmarks == ref_landmarks:
                    col.label(text="✓ Equal landmarks", icon='CHECKMARK')
                else:
                    col.label(text="⚠ Unequal landmarks", icon='ERROR')
        
        row = box.row(align=True)
        row.operator("mesh.place_landmarks_align", text="Place Landmarks")
        row.operator("mesh.delete_landmarks_align", text="Delete")
        # Region radius slider
        col = layout.column(align=True)
        col.prop(settings, "landmark_radius")
        
        row = box.row()
        row.scale_y = 1.2
        row.operator("mesh.initial_alignment", text="Perform Initial Alignment", icon='ORIENTATION_GLOBAL')
        
        # ==================== LOCAL ICP ====================
        layout.separator()
        box = layout.box()
        box.label(text="Local ICP", icon='CON_LOCLIKE')
        box.label(text="Paint regions for local alignment", icon='INFO')
        
        if settings.test_mesh and settings.reference_mesh:
            # Count painted/selected vertices
            def count_painted(mesh_obj):
                mesh = mesh_obj.data
                if "PaintSelection" not in mesh.color_attributes:
                    return 0
                color_layer = mesh.color_attributes["PaintSelection"]
                count = 0
                for color_data in color_layer.data:
                    r, g, b, a = color_data.color
                    if r < 0.95 or g < 0.95 or b < 0.95:
                        count += 1
                return count
            
            def count_selected(mesh_obj):
                return sum(1 for v in mesh_obj.data.vertices if v.select)
            
            test_painted = count_painted(settings.test_mesh)
            ref_painted = count_painted(settings.reference_mesh)
            test_selected = count_selected(settings.test_mesh)
            ref_selected = count_selected(settings.reference_mesh)
            
            # Show status (only paint/selection; landmarks are Initial Alignment only)
            col = box.column(align=True)
            col.scale_y = 0.8
            if test_selected > 0:
                col.label(text=f"Test: {test_selected} verts selected ✓", icon='CHECKMARK')
            elif test_painted > 0:
                col.label(text=f"Test: {test_painted} verts painted", icon='BRUSH_DATA')
            else:
                col.label(text="Test: No selection", icon='RADIOBUT_OFF')
            
            if ref_selected > 0:
                col.label(text=f"Ref: {ref_selected} verts selected ✓", icon='CHECKMARK')
            elif ref_painted > 0:
                col.label(text=f"Ref: {ref_painted} verts painted", icon='BRUSH_DATA')
            else:
                col.label(text="Ref: No selection", icon='RADIOBUT_OFF')
        
        # Paint workflow buttons
        box.separator()
        
        row = box.row(align=True)
        row.operator("mesh.paint_test_mesh", text="Paint Test", icon='BRUSH_DATA')
        row.operator("mesh.paint_reference_mesh", text="Paint Reference", icon='BRUSH_DATA')
        
        row = box.row()
        row.operator("mesh.accept_selection", text="Accept Selection", icon='CHECKMARK')
        
        # Local ICP button - uses painted regions only
        row = box.row()
        row.scale_y = 1.3
        row.operator("mesh.local_icp", text="Run Local ICP", icon='PLAY')
        
        # ==================== BASIC ICP SETTINGS ====================
        layout.separator()
        box = layout.box()
        box.label(text="ICP Settings", icon='SETTINGS')
        
        # Downsampling settings
        col = box.column(align=True)
        col.prop(settings, "use_downsampling")
        if settings.use_downsampling:
            col.prop(settings, "downsampling_percentage")
            
            # Show estimated vertex counts
            if settings.test_mesh and settings.reference_mesh:
                test_verts = len(settings.test_mesh.data.vertices)
                ref_verts = len(settings.reference_mesh.data.vertices)
                    
                downsampled_test = int(test_verts * (100 - settings.downsampling_percentage) / 100)
                downsampled_ref = int(ref_verts * (100 - settings.downsampling_percentage) / 100)
                
                info_col = col.column(align=True)
                info_col.scale_y = 0.7
                info_col.label(text=f"Test: {test_verts} → {downsampled_test} vertices")
                info_col.label(text=f"Ref: {ref_verts} → {downsampled_ref} vertices")
        
        # ICP parameters
        col = box.column(align=True)
        col.prop(settings, "icp_iterations")
        col.prop(settings, "outlier_percentage")
        
        # Run ICP buttons
        layout.separator()
        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator("mesh.run_basic_icp", text="Basic ICP", icon='PLAY')
        row.operator("mesh.ai_icp", text="GPU ICP", icon='SORTTIME')
        row.operator("mesh.open3d_icp", text="Open3D ICP", icon='MOD_SMOOTH')
        
        if settings.basic_alignment_complete:
            box = layout.box()
            box.label(text="✓ Basic Alignment Complete", icon='CHECKMARK')

# ===================== REGISTRATION =====================

classes = [
    MeshAlignmentSettings,
    # Dependency operators
    MESH_OT_check_dependencies,
    MESH_OT_install_open3d,
    # Basic alignment operators
    MESH_OT_select_vertices,
    MESH_OT_paint_test_mesh,
    MESH_OT_paint_reference_mesh,
    MESH_OT_accept_selection,
    MESH_OT_place_landmarks,
    MESH_OT_delete_landmarks,
    MESH_OT_pca_align,
    MESH_OT_initial_alignment,
    MESH_OT_local_icp,
    MESH_OT_ai_icp,
    MESH_OT_open3d_icp,
    MESH_OT_run_basic_icp,
    # Utility operators
    MESH_OT_fix_color_display,
    # Panels
    VIEW3D_PT_mesh_alignment_main,
    VIEW3D_PT_basic_alignment,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mesh_alignment = PointerProperty(type=MeshAlignmentSettings)
    
    # Check Open3D on startup (safely)
    try:
        is_installed, version = check_open3d_installed()
        print(f"Mesh Alignment Pro: Open3D {'available (' + version + ')' if is_installed else 'not installed'}")
    except Exception as e:
        print(f"Mesh Alignment Pro: Open3D check failed - {e}")

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.mesh_alignment

if __name__ == "__main__":
    register()
