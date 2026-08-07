bl_info = {
    "name": "Mesh Distance Analysis",
    "author": "virdentallab",
    "version": (1, 4, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Distance Analysis",
    "description": "Mesh-to-mesh signed distance and deviation analysis (Open3D raycasting)",
    "doc_url": "https://github.com/zazaven/virdentallab-blender-addons",
    "category": "Mesh",
}

import bpy
import numpy as np
from bpy.types import Panel, Operator, PropertyGroup
from bpy.props import FloatProperty, IntProperty, BoolProperty, PointerProperty, FloatVectorProperty, EnumProperty
from mathutils import Vector
from bpy_extras import view3d_utils
import bmesh
import blf
import sys
import os

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

DISTANCE_LAYER = "Distance"

# Property Group for addon settings
class MeshDistanceSettings(PropertyGroup):
    # Mesh Selection
    source_mesh: PointerProperty(
        name="Source Mesh",
        description="Source mesh for distance calculation",
        type=bpy.types.Object,
        poll=lambda self, obj: obj and obj.type == 'MESH'
    )
    
    target_mesh: PointerProperty(
        name="Target Mesh",
        description="Target mesh for distance calculation",
        type=bpy.types.Object,
        poll=lambda self, obj: obj and obj.type == 'MESH'
    )
    
    # Active direction for bidirectional mode
    active_direction: EnumProperty(
        name="Active Direction",
        description="Currently displayed direction",
        items=[
            ('SOURCE_TO_TARGET', "Source → Target", "Show Source to Target measurements"),
            ('TARGET_TO_SOURCE', "Target → Source", "Show Target to Source measurements"),
        ],
        default='SOURCE_TO_TARGET'
    )
    
    # ===== SURFACE SAMPLING SETTINGS =====
    use_surface_sampling: BoolProperty(
        name="Use Surface Sampling",
        description="Sample points from mesh surface instead of using all vertices. Useful for meshes with uneven vertex distribution",
        default=False
    )
    
    sampling_method: EnumProperty(
        name="Sampling Method",
        description="Method for sampling points from mesh surface",
        items=[
            ('UNIFORM', "Uniform", "Uniformly distributed random sampling"),
            ('POISSON_DISK', "Poisson Disk", "Blue noise sampling with minimum distance between points (recommended)"),
        ],
        default='POISSON_DISK'
    )
    
    sample_density: FloatProperty(
        name="Sample Density",
        description="Number of sample points per mm². Higher = more accurate but slower",
        default=1.0,
        min=0.01,
        max=100.0,
        precision=2
    )
    
    max_sample_points: IntProperty(
        name="Max Sample Points",
        description="Maximum number of sample points (0 = no limit)",
        default=100000,
        min=0,
        max=10000000
    )
    
    # Sample count info (read-only display)
    actual_sample_count_st: IntProperty(
        name="Sample Count S→T",
        description="Actual number of samples used for S→T",
        default=0
    )
    
    actual_sample_count_ts: IntProperty(
        name="Sample Count T→S",
        description="Actual number of samples used for T→S",
        default=0
    )
    
    use_signed_distance: BoolProperty(
        name="Signed Distance",
        description="Calculate signed distances (positive = outside, negative = inside)",
        default=True
    )
    
    # Distance Results - Source to Target
    mean_signed_st: FloatProperty(
        name="Mean Signed S→T",
        description="Mean SIGNED deviation S→T - trueness/bias, pairs with Std Dev (mm)",
        default=0.0,
        precision=4
    )

    std_deviation_st: FloatProperty(
        name="Std Dev S→T",
        description="Standard deviation of SIGNED deviations S→T - precision (mm)",
        default=0.0,
        precision=4
    )

    mad_distance_st: FloatProperty(
        name="MAD S→T",
        description="Mean ABSOLUTE distance S→T - error magnitude, do not pair with Std Dev (mm)",
        default=0.0,
        precision=4
    )
    
    max_distance_st: FloatProperty(
        name="Max Distance S→T",
        description="Maximum distance S→T (mm)",
        default=0.0,
        precision=4
    )
    
    p95_distance_st: FloatProperty(
        name="95th Percentile S→T",
        description="95th percentile of absolute distances S→T (mm)",
        default=0.0,
        precision=4
    )
    
    rms_distance_st: FloatProperty(
        name="RMS Distance S→T",
        description="Root Mean Square distance S→T (mm)",
        default=0.0,
        precision=4
    )
    
    # Distance Results - Target to Source
    mean_signed_ts: FloatProperty(
        name="Mean Signed T→S",
        description="Mean SIGNED deviation T→S - trueness/bias, pairs with Std Dev (mm)",
        default=0.0,
        precision=4
    )

    std_deviation_ts: FloatProperty(
        name="Std Dev T→S",
        description="Standard deviation of SIGNED deviations T→S - precision (mm)",
        default=0.0,
        precision=4
    )

    mad_distance_ts: FloatProperty(
        name="MAD T→S",
        description="Mean ABSOLUTE distance T→S - error magnitude, do not pair with Std Dev (mm)",
        default=0.0,
        precision=4
    )
    
    max_distance_ts: FloatProperty(
        name="Max Distance T→S",
        description="Maximum distance T→S (mm)",
        default=0.0,
        precision=4
    )
    
    p95_distance_ts: FloatProperty(
        name="95th Percentile T→S",
        description="95th percentile of absolute distances T→S (mm)",
        default=0.0,
        precision=4
    )
    
    rms_distance_ts: FloatProperty(
        name="RMS Distance T→S",
        description="Root Mean Square distance T→S (mm)",
        default=0.0,
        precision=4
    )
    
    # Combined Results (for bidirectional)
    hausdorff_distance: FloatProperty(
        name="Hausdorff Distance",
        description="Maximum distance between meshes (mm)",
        default=0.0,
        precision=4
    )
    
    hausdorff_percentile_95: FloatProperty(
        name="95th Percentile (max)",
        description="Symmetric HD95: max of the two directed 95th percentiles (mm)",
        default=0.0,
        precision=4
    )
    
    # Visualization Settings
    auto_range: BoolProperty(
        name="Auto Range",
        description="Automatically calculate min/max range",
        default=True
    )
    
    range_min: FloatProperty(
        name="Min",
        description="Minimum distance for color mapping (mm)",
        default=-0.5,
        precision=3
    )
    
    range_max: FloatProperty(
        name="Max",
        description="Maximum distance for color mapping (mm)",
        default=0.5,
        precision=3
    )
    
    # Outlier settings
    highlight_outliers: BoolProperty(
        name="Highlight Outliers",
        description="Highlight vertices outside the color range",
        default=False
    )
    
    outlier_color: FloatVectorProperty(
        name="Outlier Color",
        description="Color for outlier vertices",
        default=(1.0, 0.0, 1.0),
        min=0.0,
        max=1.0,
        subtype='COLOR',
        size=3
    )
    
    # Filter settings
    use_gaussian_filter: BoolProperty(
        name="Gaussian Filter",
        description="Apply Gaussian smoothing",
        default=True
    )
    
    gaussian_sigma: FloatProperty(
        name="Sigma",
        description="Standard deviation for Gaussian filter",
        default=2.0,
        min=0.1,
        max=10.0,
        precision=1
    )
    
    gaussian_iterations: IntProperty(
        name="Iterations",
        description="Number of smoothing iterations",
        default=1,
        min=1,
        max=10
    )
    
    use_laplacian_smooth: BoolProperty(
        name="Laplacian Smoothing",
        description="Apply Laplacian smoothing",
        default=True
    )
    
    laplacian_factor: FloatProperty(
        name="Factor",
        description="Laplacian smoothing factor",
        default=0.5,
        min=0.0,
        max=1.0,
        precision=2
    )
    
    laplacian_iterations: IntProperty(
        name="Iterations",
        description="Number of Laplacian iterations",
        default=1,
        min=1,
        max=10
    )
    
    # Display settings
    show_legend: BoolProperty(default=False)
    show_histogram: BoolProperty(default=False)
    show_probe: BoolProperty(default=False)
    
    # Status flags
    distance_calculation_complete: BoolProperty(default=False)
    open3d_available: BoolProperty(default=False)

# OPERATORS

class MESH_OT_check_open3d_distance(Operator):
    """Check if Open3D is installed"""
    bl_idname = "mesh.check_open3d_distance"
    bl_label = "Check Open3D"
    
    def execute(self, context):
        try:
            import open3d as o3d
            version = o3d.__version__
            self.report({'INFO'}, f"Open3D {version} is installed")
            context.scene.mesh_distance.open3d_available = True
        except ImportError:
            self.report({'ERROR'}, "Open3D not found. Install with: pip install open3d")
            context.scene.mesh_distance.open3d_available = False
        return {'FINISHED'}

def alignment_meshes(context):
    """Return (test_mesh, reference_mesh) from the Mesh Alignment addon.

    Mesh Alignment stores its pair on scene.mesh_alignment. That addon may not
    be installed or enabled, so every access is guarded and a missing pair just
    yields (None, None) rather than an error.
    """
    align = getattr(context.scene, "mesh_alignment", None)
    if align is None:
        return None, None
    test = getattr(align, "test_mesh", None)
    ref = getattr(align, "reference_mesh", None)
    if test is not None and getattr(test, "type", None) != 'MESH':
        test = None
    if ref is not None and getattr(ref, "type", None) != 'MESH':
        ref = None
    return test, ref


class MESH_OT_use_alignment_meshes(Operator):
    """Load the mesh pair from the Mesh Alignment addon (test -> Source, reference -> Target)"""
    bl_idname = "mesh.use_alignment_meshes"
    bl_label = "Use Mesh Alignment Pair"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        test, ref = alignment_meshes(context)
        return bool(test and ref)

    def execute(self, context):
        settings = context.scene.mesh_distance
        test, ref = alignment_meshes(context)
        if not (test and ref):
            self.report({'ERROR'}, "Mesh Alignment has no test/reference pair set")
            return {'CANCELLED'}
        settings.source_mesh = test
        settings.target_mesh = ref
        self.report({'INFO'}, f"Loaded {test.name} -> {ref.name} from Mesh Alignment")
        return {'FINISHED'}


class MESH_OT_calculate_mesh_distances(Operator):
    """Calculate distances between two meshes using Open3D RayCasting"""
    bl_idname = "mesh.calculate_mesh_distances"
    bl_label = "Calculate Distances"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        settings = context.scene.mesh_distance
        if (settings.source_mesh and settings.target_mesh and
                settings.source_mesh.type == 'MESH' and
                settings.target_mesh.type == 'MESH'):
            return True
        # Nothing picked yet, but Mesh Alignment has a pair we can fall back on
        test, ref = alignment_meshes(context)
        return bool(test and ref)
    
    def calculate_one_direction(self, from_obj, to_obj, settings, for_stats=False):
        """Calculate distances from one mesh to another
        
        Args:
            from_obj: Source mesh object
            to_obj: Target mesh object  
            settings: Addon settings
            for_stats: If True and sampling enabled, use sampled points for statistics
                      If False, always use vertices for visualization
        """
        try:
            import open3d as o3d
        except ImportError:
            return None, 0
        
        # Convert Blender mesh to Open3D triangle mesh
        def blender_to_o3d_mesh(obj):
            mesh = obj.data
            vertices = np.array([obj.matrix_world @ v.co for v in mesh.vertices])
            
            # Get triangles (support ngons)
            triangles = []
            for poly in mesh.polygons:
                if len(poly.vertices) == 3:
                    triangles.append(list(poly.vertices))
                elif len(poly.vertices) == 4:
                    v = poly.vertices
                    triangles.append([v[0], v[1], v[2]])
                    triangles.append([v[0], v[2], v[3]])
                elif len(poly.vertices) > 4:
                    # Fan triangulation for ngons
                    v = poly.vertices
                    for i in range(1, len(v) - 1):
                        triangles.append([v[0], v[i], v[i+1]])
            
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
            o3d_mesh.compute_vertex_normals()
            
            return o3d_mesh
        
        # Create Open3D meshes
        from_o3d = blender_to_o3d_mesh(from_obj)
        to_o3d = blender_to_o3d_mesh(to_obj)
        
        # Determine query points
        if for_stats and settings.use_surface_sampling:
            # For statistics: use sampled points (accurate, unbiased)
            surface_area = from_o3d.get_surface_area()
            target_points = int(surface_area * settings.sample_density)
            
            if settings.max_sample_points > 0:
                target_points = min(target_points, settings.max_sample_points)
            
            target_points = max(100, target_points)
            
            if settings.sampling_method == 'UNIFORM':
                pcd = from_o3d.sample_points_uniformly(number_of_points=target_points)
            else:  # POISSON_DISK
                pcd = from_o3d.sample_points_poisson_disk(number_of_points=target_points, init_factor=5)
            
            query_points_np = np.asarray(pcd.points)
            sample_count = len(query_points_np)
        else:
            # For visualization: use vertices (sharp, detailed)
            query_points_np = np.asarray(from_o3d.vertices)
            sample_count = len(query_points_np)
        
        # Create RayCasting scene with target mesh
        scene = o3d.t.geometry.RaycastingScene()
        to_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(to_o3d)
        scene.add_triangles(to_mesh_t)
        
        # Compute signed distances
        query_points = o3d.core.Tensor(query_points_np.astype(np.float32), dtype=o3d.core.Dtype.Float32)
        signed_distances = scene.compute_signed_distance(query_points)
        distances = signed_distances.numpy()
        
        # If not using signed distance, take absolute values
        if not settings.use_signed_distance:
            distances = np.abs(distances)
        
        return distances, sample_count
    
    def execute(self, context):
        settings = context.scene.mesh_distance

        # Fall back to the Mesh Alignment pair so the same two meshes do not
        # have to be picked again after aligning them
        if not (settings.source_mesh and settings.target_mesh):
            test, ref = alignment_meshes(context)
            if test and ref:
                settings.source_mesh = test
                settings.target_mesh = ref
                self.report({'INFO'}, f"Using Mesh Alignment pair: {test.name} -> {ref.name}")

        source_obj = settings.source_mesh
        target_obj = settings.target_mesh
        if not (source_obj and target_obj):
            self.report({'ERROR'}, "Pick a source and target mesh first")
            return {'CANCELLED'}
        
        # === VISUALIZATION: Always vertex-based (sharp deviation display) ===
        distances_st_viz, _ = self.calculate_one_direction(source_obj, target_obj, settings, for_stats=False)
        distances_ts_viz, _ = self.calculate_one_direction(target_obj, source_obj, settings, for_stats=False)
        
        if distances_st_viz is None or distances_ts_viz is None:
            self.report({'ERROR'}, "Open3D not installed")
            return {'CANCELLED'}
        
        # === STATISTICS: Sampled if enabled (accurate, unbiased) ===
        if settings.use_surface_sampling:
            distances_st_stats, count_st = self.calculate_one_direction(source_obj, target_obj, settings, for_stats=True)
            distances_ts_stats, count_ts = self.calculate_one_direction(target_obj, source_obj, settings, for_stats=True)
        else:
            distances_st_stats = distances_st_viz
            distances_ts_stats = distances_ts_viz
            count_st = len(distances_st_viz)
            count_ts = len(distances_ts_viz)
        
        # Store sample counts
        settings.actual_sample_count_st = count_st
        settings.actual_sample_count_ts = count_ts
        
        # Calculate statistics from SAMPLED data (accurate)
        # Signed mean + signed SD describe the SAME distribution, so they are the
        # pair that may be reported as "mean +/- SD" (trueness +/- precision).
        # MAD is the mean of the ABSOLUTE values - a different distribution, so it
        # must never be shown with the signed SD next to it.
        abs_distances_st = np.abs(distances_st_stats)
        settings.mean_signed_st = float(np.mean(distances_st_stats))
        settings.std_deviation_st = float(np.std(distances_st_stats))
        settings.mad_distance_st = float(np.mean(abs_distances_st))
        settings.max_distance_st = float(np.max(abs_distances_st))
        settings.rms_distance_st = float(np.sqrt(np.mean(distances_st_stats**2)))
        settings.p95_distance_st = float(np.percentile(abs_distances_st, 95))
        
        abs_distances_ts = np.abs(distances_ts_stats)
        settings.mean_signed_ts = float(np.mean(distances_ts_stats))
        settings.std_deviation_ts = float(np.std(distances_ts_stats))
        settings.mad_distance_ts = float(np.mean(abs_distances_ts))
        settings.max_distance_ts = float(np.max(abs_distances_ts))
        settings.rms_distance_ts = float(np.sqrt(np.mean(distances_ts_stats**2)))
        settings.p95_distance_ts = float(np.percentile(abs_distances_ts, 95))
        
        # Symmetric (two-way) figures. Both are the max of the two directed
        # values, matching the standard definitions: Hausdorff is
        # max(h(S->T), h(T->S)) and HD95 is max(P95(S->T), P95(T->S)).
        # NOTE: do NOT pool the two directions and take one percentile - that
        # weights whichever direction has more samples/vertices and is not the
        # HD95 reported in the literature.
        settings.hausdorff_distance = float(max(np.max(abs_distances_st), np.max(abs_distances_ts)))
        settings.hausdorff_percentile_95 = float(max(settings.p95_distance_st,
                                                     settings.p95_distance_ts))
        
        # Store VERTEX-BASED distances for visualization (sharp)
        context.scene['mesh_distances_st'] = distances_st_viz.tolist()
        context.scene['mesh_distances_ts'] = distances_ts_viz.tolist()
        
        # Store SAMPLED distances for histogram (accurate distribution)
        context.scene['mesh_distances_st_raw'] = distances_st_stats.tolist()
        context.scene['mesh_distances_ts_raw'] = distances_ts_stats.tolist()
        
        # Set active direction distances
        if settings.active_direction == 'SOURCE_TO_TARGET':
            context.scene['mesh_distances'] = distances_st_viz.tolist()
            context.scene['mesh_distances_raw'] = distances_st_stats.tolist()
            context.scene['source_mesh_name'] = source_obj.name
        else:
            context.scene['mesh_distances'] = distances_ts_viz.tolist()
            context.scene['mesh_distances_raw'] = distances_ts_stats.tolist()
            context.scene['source_mesh_name'] = target_obj.name
        
        context.scene['current_direction'] = settings.active_direction
        
        settings.distance_calculation_complete = True
        
        # Automatically apply color mapping after calculation
        bpy.ops.mesh.visualize_mesh_distances()
        
        if settings.use_surface_sampling:
            self.report({'INFO'}, f"Hybrid mode: Stats from {count_st:,}+{count_ts:,} samples, Viz from vertices")
        else:
            self.report({'INFO'}, "Distance calculation complete")
        
        return {'FINISHED'}

class MESH_OT_switch_direction(Operator):
    """Switch between measurement directions (for bidirectional mode)"""
    bl_idname = "mesh.switch_direction"
    bl_label = "Switch Direction"
    bl_options = {'REGISTER', 'UNDO'}
    
    direction: EnumProperty(
        items=[
            ('SOURCE_TO_TARGET', "Source → Target", ""),
            ('TARGET_TO_SOURCE', "Target → Source", ""),
        ],
        default='SOURCE_TO_TARGET'
    )
    
    @classmethod
    def poll(cls, context):
        settings = context.scene.mesh_distance
        return settings.distance_calculation_complete
    
    def execute(self, context):
        settings = context.scene.mesh_distance

        # Pick the stored arrays for the requested direction
        if self.direction == 'SOURCE_TO_TARGET':
            distances = context.scene.get('mesh_distances_st')
            distances_raw = context.scene.get('mesh_distances_st_raw')
            mesh_obj = settings.source_mesh
        else:
            distances = context.scene.get('mesh_distances_ts')
            distances_raw = context.scene.get('mesh_distances_ts_raw')
            mesh_obj = settings.target_mesh

        # Validate BEFORE mutating any state, so a failed switch leaves the
        # scene consistent instead of half-updated
        if distances is None or len(distances) == 0:
            self.report({'ERROR'}, "No stored data for that direction - recalculate first")
            return {'CANCELLED'}
        if mesh_obj is None:
            self.report({'ERROR'}, "Mesh for that direction is no longer assigned")
            return {'CANCELLED'}
        if len(distances) != len(mesh_obj.data.vertices):
            self.report({'ERROR'},
                        f"Stored data ({len(distances)}) does not match {mesh_obj.name} "
                        f"({len(mesh_obj.data.vertices)} verts) - recalculate first")
            return {'CANCELLED'}

        settings.active_direction = self.direction
        context.scene['source_mesh_name'] = mesh_obj.name

        # NOTE: assigning an IDPropertyArray straight onto an existing key fills
        # that array IN PLACE and keeps its old length, silently truncating it or
        # leaving a stale tail. The two directions differ in length whenever the
        # meshes differ in vertex/sample count, so wrap in list() to force the
        # property to be replaced rather than filled.
        context.scene['mesh_distances'] = list(distances)
        context.scene['mesh_distances_raw'] = list(distances_raw) if distances_raw else list(distances)
        context.scene['current_direction'] = self.direction

        # Re-visualize with new direction
        bpy.ops.mesh.visualize_mesh_distances()

        self.report({'INFO'}, f"Switched to {self.direction.replace('_', ' ').title()}")
        return {'FINISHED'}

class MESH_OT_visualize_distances(Operator):
    """Visualize distance results with color mapping"""
    bl_idname = "mesh.visualize_mesh_distances"
    bl_label = "Visualize Distances"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        settings = context.scene.mesh_distance
        return settings.distance_calculation_complete
    
    def get_professional_colors(self, values, min_val, max_val):
        """Vectorised color mapping (Blue-Cyan-Green-Yellow-Orange-Red).

        Array equivalent of the former per-vertex scalar helper: identical
        breakpoints and blends, but evaluated over the whole distance array
        at once instead of once per mesh loop.
        """
        values = np.asarray(values, dtype=np.float64)

        # Normalize
        if max_val > min_val:
            t = (values - min_val) / (max_val - min_val)
        else:
            t = np.full(values.shape, 0.5)
        t = np.clip(t, 0.0, 1.0)

        r = np.empty_like(t)
        g = np.empty_like(t)
        b = np.empty_like(t)

        m = t <= 0.2
        blend = t[m] / 0.2
        r[m] = 0.0
        g[m] = blend * 0.5
        b[m] = 1.0

        m = (t > 0.2) & (t <= 0.4)
        blend = (t[m] - 0.2) / 0.2
        r[m] = 0.0
        g[m] = 0.5 + blend * 0.5
        b[m] = 1.0 - blend

        m = (t > 0.4) & (t <= 0.6)
        blend = (t[m] - 0.4) / 0.2
        r[m] = blend
        g[m] = 1.0
        b[m] = 0.0

        m = (t > 0.6) & (t <= 0.8)
        blend = (t[m] - 0.6) / 0.2
        r[m] = 1.0
        g[m] = 1.0 - blend * 0.5
        b[m] = 0.0

        m = t > 0.8
        blend = (t[m] - 0.8) / 0.2
        r[m] = 1.0
        g[m] = 0.5 - blend * 0.5
        b[m] = 0.0

        return np.stack([r, g, b, np.ones_like(t)], axis=1).astype(np.float32)

    def apply_gaussian_filter(self, distances, mesh_obj, sigma, iterations):
        """Apply Gaussian smoothing to distance values"""
        if sigma <= 0 or iterations <= 0:
            return distances
        
        # Build vertex neighbor map
        bm = bmesh.new()
        bm.from_mesh(mesh_obj.data)
        bm.verts.ensure_lookup_table()
        
        neighbors = {}
        for vert in bm.verts:
            vert_neighbors = []
            for edge in vert.link_edges:
                other_vert = edge.other_vert(vert)
                distance = (vert.co - other_vert.co).length
                vert_neighbors.append((other_vert.index, distance))
            neighbors[vert.index] = vert_neighbors
        
        bm.free()
        
        # Apply Gaussian filter
        smoothed = distances.copy()
        
        for _ in range(iterations):
            new_distances = smoothed.copy()
            
            for i, dist in enumerate(smoothed):
                if i in neighbors:
                    weighted_sum = dist
                    weight_sum = 1.0
                    
                    for n_idx, edge_dist in neighbors[i]:
                        if n_idx < len(smoothed):
                            weight = np.exp(-(edge_dist * edge_dist) / (2 * sigma * sigma))
                            weighted_sum += smoothed[n_idx] * weight
                            weight_sum += weight
                    
                    new_distances[i] = weighted_sum / weight_sum
            
            smoothed = new_distances
        
        return smoothed
    
    def apply_laplacian_smooth(self, distances, mesh_obj, factor, iterations):
        """Apply Laplacian smoothing"""
        if factor <= 0 or iterations <= 0:
            return distances
        
        # Build vertex neighbor map
        bm = bmesh.new()
        bm.from_mesh(mesh_obj.data)
        bm.verts.ensure_lookup_table()
        
        neighbors = {}
        for vert in bm.verts:
            neighbors[vert.index] = [e.other_vert(vert).index for e in vert.link_edges]
        
        bm.free()
        
        # Apply Laplacian smoothing
        smoothed = distances.copy()
        
        for _ in range(iterations):
            new_distances = smoothed.copy()
            
            for i, dist in enumerate(smoothed):
                if i in neighbors and neighbors[i]:
                    neighbor_avg = np.mean([smoothed[n] for n in neighbors[i] if n < len(smoothed)])
                    new_distances[i] = (1 - factor) * dist + factor * neighbor_avg
            
            smoothed = new_distances
        
        return smoothed
    
    def execute(self, context):
        settings = context.scene.mesh_distance
        
        # Get distances and mesh
        distances = context.scene.get('mesh_distances')
        source_mesh_name = context.scene.get('source_mesh_name')
        
        if not distances or not source_mesh_name:
            self.report({'ERROR'}, "No distance data found")
            return {'CANCELLED'}
        
        source_obj = bpy.data.objects.get(source_mesh_name)
        if not source_obj:
            self.report({'ERROR'}, f"Mesh {source_mesh_name} not found")
            return {'CANCELLED'}
        
        distances = np.array(distances)
        
        # Apply filters
        if settings.use_gaussian_filter:
            distances = self.apply_gaussian_filter(
                distances, source_obj, 
                settings.gaussian_sigma, 
                settings.gaussian_iterations
            )
        
        if settings.use_laplacian_smooth:
            distances = self.apply_laplacian_smooth(
                distances, source_obj,
                settings.laplacian_factor,
                settings.laplacian_iterations
            )
        
        # Determine range
        if settings.auto_range:
            abs_distances = np.abs(distances)
            percentile_val = np.percentile(abs_distances, 95)
            settings.range_min = -percentile_val
            settings.range_max = percentile_val
        
        # Get (or create) our own colour attribute.
        # Do NOT rely on mesh.vertex_colors.active: that legacy collection only
        # exposes CORNER/BYTE_COLOR attributes, and .active follows
        # color_attributes.active_color. Another addon can leave a POINT/FLOAT
        # attribute active, or remove the active one without setting a new one
        # (Mesh Alignment's PaintSelection cleanup does exactly that), and then
        # .active is None even though our layer exists.
        mesh = source_obj.data
        color_layer = mesh.color_attributes.get(DISTANCE_LAYER)
        if color_layer is None:
            color_layer = mesh.color_attributes.new(
                name=DISTANCE_LAYER, type='BYTE_COLOR', domain='CORNER'
            )
        # Make it active so SOLID + VERTEX shading actually displays it
        mesh.color_attributes.active_color = color_layer
        mesh.color_attributes.render_color_index = mesh.color_attributes.active_color_index

        # Per-vertex colors, computed in a single vectorised pass
        vert_colors = self.get_professional_colors(
            distances, settings.range_min, settings.range_max
        )
        if settings.highlight_outliers:
            outliers = (distances < settings.range_min) | (distances > settings.range_max)
            vert_colors[outliers] = (*settings.outlier_color, 1.0)

        # Write the colours in one foreach_set instead of walking faces/loops in
        # Python. Existing colours are read back first so elements whose vertex
        # has no distance value keep what they had (old per-loop index guard).
        # Both domains are handled: an attribute created by an older version of
        # this addon - or by another tool - may be POINT instead of CORNER.
        if color_layer.domain == 'POINT':
            n_elems = len(mesh.vertices)
            flat = np.empty(n_elems * 4, dtype=np.float32)
            color_layer.data.foreach_get("color", flat)
            elem_colors = flat.reshape(-1, 4)
            n = min(n_elems, len(vert_colors))
            elem_colors[:n] = vert_colors[:n]
        else:
            n_loops = len(mesh.loops)
            loop_vidx = np.empty(n_loops, dtype=np.int32)
            mesh.loops.foreach_get("vertex_index", loop_vidx)

            flat = np.empty(n_loops * 4, dtype=np.float32)
            color_layer.data.foreach_get("color", flat)
            loop_colors = flat.reshape(-1, 4)

            valid = loop_vidx < len(vert_colors)
            loop_colors[valid] = vert_colors[loop_vidx[valid]]

        color_layer.data.foreach_set("color", flat)
        mesh.update()
        # foreach_set writes straight into mesh data and, unlike bmesh.to_mesh(),
        # does not flag the object for re-evaluation - without these tags the
        # viewport can keep showing the previous direction's colors
        mesh.update_tag()
        source_obj.update_tag()


        # Set viewport shading
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'SOLID'
                        space.shading.color_type = 'VERTEX'
                area.tag_redraw()

        # Show whichever mesh carries the colors for the active direction and
        # hide its counterpart, so switching direction visibly swaps the map.
        # Both hide flags matter: hide_viewport is the monitor icon, hide_get()
        # is the eye icon, and either one on its own keeps the mesh invisible.
        counterpart = None
        if source_obj == settings.source_mesh:
            counterpart = settings.target_mesh
        elif source_obj == settings.target_mesh:
            counterpart = settings.source_mesh

        source_obj.hide_viewport = False
        source_obj.hide_set(False)
        if counterpart is not None and counterpart != source_obj:
            counterpart.hide_set(True)

        # Make source object active
        bpy.ops.object.select_all(action='DESELECT')
        source_obj.select_set(True)
        context.view_layer.objects.active = source_obj
        
        self.report({'INFO'}, f"Visualization complete ({len(distances)} vertices)")
        return {'FINISHED'}

class MESH_OT_apply_color_mapping_distance(Operator):
    """Apply color mapping to mesh"""
    bl_idname = "mesh.apply_color_mapping_distance"
    bl_label = "Update Color Mapping"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        return bpy.ops.mesh.visualize_mesh_distances()

# GPU Drawing Functions (if available)
if GPU_AVAILABLE:
    # Position and scale for drag and drop
    legend_position = {'x': 50, 'y': 100}
    legend_scale = {'factor': 1.0}
    histogram_position = {'x': 50, 'y': 50}
    histogram_scale = {'factor': 1.0}
    
    # Distance probe data
    probe_data = {
        'active': False,
        'position': (0, 0),
        'distance': 0.0,
        'vertex_index': -1,
        'world_pos': Vector((0, 0, 0))
    }
    
    # Saved measurements
    saved_measurements = []
    
    def draw_color_legend(context):
        """Draw color scale legend with drag and scale support"""
        settings = context.scene.mesh_distance
        
        scale = legend_scale['factor']
        x = legend_position['x']
        y = context.area.height - legend_position['y']
        width = 60 * scale  # Genişlik 60 piksel olarak ayarlandı
        height = 500 * scale  # Uzunluk 500 piksel
        
        # İçerik için sağa kayma offset'i
        content_offset = 10 * scale
        
        # Get current direction for title
        current_dir = context.scene.get('current_direction', 'SOURCE_TO_TARGET')
        if current_dir == 'SOURCE_TO_TARGET':
            dir_text = "S→T"
        else:
            dir_text = "T→S"
        
        # Background
        bg_padding = 15 * scale
        vertices = [
            (x - bg_padding, y + 60 * scale),
            (x + width + 110 * scale, y + 60 * scale),  # Değer label'ları için genişlik
            (x + width + 110 * scale, y - height - 60 * scale),
            (x - bg_padding, y - height - 60 * scale)
        ]
        
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'TRI_FAN', {"pos": vertices})
        shader.bind()
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.95))
        batch.draw(shader)
        
        # Border
        border_batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": vertices})
        shader.uniform_float("color", (0.2, 0.2, 0.2, 1))
        border_batch.draw(shader)
        
        # Direction indicator - sağa kaydırıldı
        blf.size(0, int(18 * scale))
        blf.color(0, 0.2, 0.2, 0.8, 1)
        blf.position(0, x + content_offset + width/2 - 15 * scale, y + 40 * scale, 0)
        blf.draw(0, dir_text)
        
        # Draw gradient - sağa kaydırıldı
        vertices = []
        colors = []
        indices = []
        
        steps = 50
        for i in range(steps):
            t = i / (steps - 1)
            ypos = y - height * t
            
            # Professional color
            t_rev = 1.0 - t
            if t_rev <= 0.2:
                blend = t_rev / 0.2
                color = (0.0, blend * 0.5, 1.0, 1.0)
            elif t_rev <= 0.4:
                blend = (t_rev - 0.2) / 0.2
                color = (0.0, 0.5 + blend * 0.5, 1.0 - blend, 1.0)
            elif t_rev <= 0.6:
                blend = (t_rev - 0.4) / 0.2
                color = (blend, 1.0, 0.0, 1.0)
            elif t_rev <= 0.8:
                blend = (t_rev - 0.6) / 0.2
                color = (1.0, 1.0 - blend * 0.5, 0.0, 1.0)
            else:
                blend = (t_rev - 0.8) / 0.2
                color = (1.0, 0.5 - blend * 0.5, 0.0, 1.0)
            
            vertices.extend([(x + content_offset, ypos), (x + content_offset + width, ypos)])
            colors.extend([color, color])
            
            if i < steps - 1:
                base = i * 2
                indices.extend([
                    (base, base + 1, base + 3),
                    (base, base + 3, base + 2)
                ])
        
        # Draw gradient
        shader = gpu.shader.from_builtin('FLAT_COLOR')
        batch = batch_for_shader(
            shader, 'TRIS',
            {"pos": vertices, "color": colors},
            indices=indices
        )
        shader.bind()
        batch.draw(shader)
        
        # Draw border - sağa kaydırıldı
        border_vertices = [
            (x + content_offset, y), (x + content_offset + width, y),
            (x + content_offset + width, y - height), (x + content_offset, y - height),
            (x + content_offset, y)
        ]
        
        border_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        border_batch = batch_for_shader(
            border_shader, 'LINE_STRIP',
            {"pos": border_vertices}
        )
        border_shader.bind()
        border_shader.uniform_float("color", (0, 0, 0, 1))
        border_batch.draw(border_shader)
        
        # Text labels
        blf.size(0, int(16 * scale))
        blf.color(0, 0, 0, 0, 1)
        
        # Title - sağa kaydırıldı
        blf.position(0, x + content_offset + width/2 - 15 * scale, y + 20 * scale, 0)
        blf.draw(0, "mm")
        
        # Values - 7 adet değer göster - sağa kaydırıldı
        blf.size(0, int(14 * scale))
        num_labels = 7
        for i in range(num_labels):
            t = i / (num_labels - 1)
            label_y = y - (height * t)
            value = settings.range_max - t * (settings.range_max - settings.range_min)
            
            # Draw tick mark - sağa kaydırıldı
            tick_vertices = [
                (x + content_offset + width, label_y),
                (x + content_offset + width + 5 * scale, label_y)
            ]
            tick_batch = batch_for_shader(border_shader, 'LINES', {"pos": tick_vertices})
            border_shader.uniform_float("color", (0.3, 0.3, 0.3, 1))
            tick_batch.draw(border_shader)
            
            # Draw value - sağa kaydırıldı
            blf.position(0, x + content_offset + width + 10 * scale, label_y - 6 * scale, 0)
            blf.draw(0, f"{value:.3f}")
        
        # Bottom title - sağa kaydırıldı
        blf.size(0, int(18 * scale))
        blf.position(0, x + content_offset + width/2 - 45 * scale, y - height - 45 * scale, 0)
        blf.draw(0, "Color Scale")
    
    def draw_distance_histogram(context):
        """Draw distance histogram with professional style"""
        # Use raw data for histogram (sampled points, not interpolated)
        distances = context.scene.get('mesh_distances_raw')
        if not distances:
            distances = context.scene.get('mesh_distances')
        if not distances:
            return
        
        distances = np.array(distances)
        settings = context.scene.mesh_distance
        
        # Get current direction
        current_dir = context.scene.get('current_direction', 'SOURCE_TO_TARGET')
        if current_dir == 'SOURCE_TO_TARGET':
            mean = settings.mean_signed_st
            std = settings.std_deviation_st
            sample_count = settings.actual_sample_count_st
            dir_text = "Source → Target"
        else:
            mean = settings.mean_signed_ts
            std = settings.std_deviation_ts
            sample_count = settings.actual_sample_count_ts
            dir_text = "Target → Source"
        
        scale = histogram_scale['factor']
        x = histogram_position['x']
        y = context.area.height - histogram_position['y']
        width = 800 * scale
        height = 600 * scale
        
        # Background
        vertices = [
            (x, y), (x + width, y),
            (x + width, y - height), (x, y - height)
        ]
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'TRI_FAN', {"pos": vertices})
        shader.bind()
        shader.uniform_float("color", (0.15, 0.15, 0.15, 0.95))
        batch.draw(shader)
        
        # Border
        border_batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": vertices})
        shader.uniform_float("color", (0.4, 0.4, 0.4, 1))
        border_batch.draw(shader)
        
        # Title with direction
        blf.size(0, int(20 * scale))
        blf.color(0, 1, 1, 1, 1)
        blf.position(0, x + width/2 - 150 * scale, y - 30 * scale, 0)
        blf.draw(0, f"Distance Histogram - {dir_text}")
        
        # Statistics (with sample count if sampling was used)
        blf.size(0, int(16 * scale))
        stats_y = y - 60 * scale
        blf.position(0, x + width/2 - 200 * scale, stats_y, 0)
        if settings.use_surface_sampling and sample_count > 0:
            blf.draw(0, f"μ = {mean:.4f}    σ = {std:.4f}    n = {sample_count:,}")
        else:
            blf.draw(0, f"μ = {mean:.4f}    σ = {std:.4f}")
        
        # Create histogram
        hist, bins = np.histogram(distances, bins=100, range=(settings.range_min, settings.range_max))
        max_count = hist.max() if hist.max() > 0 else 1
        
        # Graph area
        graph_left = x + 100 * scale
        graph_right = x + width - 80 * scale
        graph_bottom = y - height + 100 * scale
        graph_top = y - 120 * scale
        graph_width = graph_right - graph_left
        graph_height = graph_top - graph_bottom
        
        # Draw histogram bars with colors
        bar_width = graph_width / len(hist)
        
        for i, count in enumerate(hist):
            if count > 0:
                bar_x = graph_left + i * bar_width
                bar_height = (count / max_count) * graph_height
                
                # Get color based on bin value
                bin_value = (bins[i] + bins[i+1]) / 2
                
                # Professional color
                if settings.range_max > settings.range_min:
                    t = (bin_value - settings.range_min) / (settings.range_max - settings.range_min)
                else:
                    t = 0.5
                t = max(0.0, min(1.0, t))
                
                t_rev = 1.0 - t
                if t_rev <= 0.2:
                    blend = t_rev / 0.2
                    color = (0.0, blend * 0.5, 1.0, 0.9)
                elif t_rev <= 0.4:
                    blend = (t_rev - 0.2) / 0.2
                    color = (0.0, 0.5 + blend * 0.5, 1.0 - blend, 0.9)
                elif t_rev <= 0.6:
                    blend = (t_rev - 0.4) / 0.2
                    color = (blend, 1.0, 0.0, 0.9)
                elif t_rev <= 0.8:
                    blend = (t_rev - 0.6) / 0.2
                    color = (1.0, 1.0 - blend * 0.5, 0.0, 0.9)
                else:
                    blend = (t_rev - 0.8) / 0.2
                    color = (1.0, 0.5 - blend * 0.5, 0.0, 0.9)
                
                # Draw bar
                bar_vertices = [
                    (bar_x, graph_bottom),
                    (bar_x + bar_width * 0.9, graph_bottom),
                    (bar_x + bar_width * 0.9, graph_bottom + bar_height),
                    (bar_x, graph_bottom + bar_height)
                ]
                
                bar_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
                bar_batch = batch_for_shader(bar_shader, 'TRI_FAN', {"pos": bar_vertices})
                bar_shader.bind()
                bar_shader.uniform_float("color", color)
                bar_batch.draw(bar_shader)
        
        # Draw axes
        axes_vertices = [
            (graph_left, graph_bottom), (graph_right, graph_bottom),
            (graph_left, graph_bottom), (graph_left, graph_top)
        ]
        axes_batch = batch_for_shader(shader, 'LINES', {"pos": axes_vertices})
        shader.uniform_float("color", (0.7, 0.7, 0.7, 1))
        axes_batch.draw(shader)
        
        # X-axis labels
        blf.size(0, int(14 * scale))
        blf.color(0, 1, 1, 1, 1)
        num_x_labels = 7
        for i in range(num_x_labels):
            t = i / (num_x_labels - 1)
            label_x = graph_left + t * graph_width
            value = settings.range_min + t * (settings.range_max - settings.range_min)
            
            # Draw tick
            tick_vertices = [(label_x, graph_bottom), (label_x, graph_bottom - 5 * scale)]
            tick_batch = batch_for_shader(shader, 'LINES', {"pos": tick_vertices})
            tick_batch.draw(shader)
            
            # Draw label
            label_text = f"{value:.2f}"
            text_width = len(label_text) * 7 * scale
            blf.position(0, label_x - text_width/2, graph_bottom - 25 * scale, 0)
            blf.draw(0, label_text)
        
        # Y-axis labels
        num_y_labels = 6
        for i in range(num_y_labels):
            t = i / (num_y_labels - 1)
            label_y = graph_bottom + t * graph_height
            value = int(t * max_count)
            
            # Draw tick
            tick_vertices = [(graph_left, label_y), (graph_left - 5 * scale, label_y)]
            tick_batch = batch_for_shader(shader, 'LINES', {"pos": tick_vertices})
            tick_batch.draw(shader)
            
            # Draw label
            label_text = f"{value}"
            text_width = len(label_text) * 7 * scale
            blf.position(0, graph_left - text_width - 10 * scale, label_y - 5 * scale, 0)
            blf.draw(0, label_text)
        
        # Axis labels
        blf.size(0, int(16 * scale))
        blf.position(0, graph_left + graph_width/2 - 50 * scale, graph_bottom - 50 * scale, 0)
        blf.draw(0, "Distance (mm)")
        
        blf.position(0, x + 30 * scale, graph_bottom + graph_height/2 - 5 * scale, 0)
        blf.draw(0, "Count")
    
    def get_view_scale(context, world_pos):
        """Get scale factor based on view distance - simple and fast"""
        rv3d = context.region_data
        if rv3d is None:
            return 1.0
        
        view_loc = rv3d.view_matrix.inverted().translation
        dist = (world_pos - view_loc).length
        
        # Scale: closer = bigger, reference at 100 units
        if dist > 0:
            scale = 100.0 / dist
        else:
            scale = 1.0
        
        return max(0.4, min(2.5, scale))
    
    def draw_distance_probe(context):
        """Draw distance probe information like GOM Inspect"""
        settings = context.scene.mesh_distance
        
        # Get current direction
        current_dir = context.scene.get('current_direction', 'SOURCE_TO_TARGET')
        if current_dir == 'SOURCE_TO_TARGET':
            mean = settings.mean_signed_st
            std = settings.std_deviation_st
            dir_text = "S→T"
        else:
            mean = settings.mean_signed_ts
            std = settings.std_deviation_ts
            dir_text = "T→S"
        
        # Get source mesh for world position updates
        source_mesh_name = context.scene.get('source_mesh_name')
        source_obj = None
        if source_mesh_name:
            source_obj = bpy.data.objects.get(source_mesh_name)
        
        region = context.region
        rv3d = context.region_data
        
        # Draw saved measurements first
        for i, measurement in enumerate(saved_measurements):
            distance = measurement['distance']
            vertex_idx = measurement['vertex_index']
            measurement_dir = measurement.get('direction', 'SOURCE_TO_TARGET')
            
            # Update screen position from 3D world position
            if source_obj and vertex_idx < len(source_obj.data.vertices):
                world_pos = source_obj.matrix_world @ source_obj.data.vertices[vertex_idx].co
                screen_pos = view3d_utils.location_3d_to_region_2d(region, rv3d, world_pos)
                
                if screen_pos:
                    mx, my = screen_pos
                else:
                    continue
            else:
                continue
            
            # Get zoom scale
            scale = get_view_scale(context, world_pos)
            
            # Scaled dimensions
            label_width = 140 * scale
            label_height = 45 * scale
            offset_y = 20 * scale
            
            vertices = [
                (mx - label_width/2, my + offset_y),
                (mx + label_width/2, my + offset_y),
                (mx + label_width/2, my + offset_y + label_height),
                (mx - label_width/2, my + offset_y + label_height)
            ]
            
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            batch = batch_for_shader(shader, 'TRI_FAN', {"pos": vertices})
            shader.bind()
            
            bg_color = (0.85, 0.85, 0.85, 0.9)
            shader.uniform_float("color", bg_color)
            batch.draw(shader)
            
            # Draw border based on deviation
            abs_dist = abs(distance)
            if abs_dist <= abs(mean):
                border_color = (0.2, 0.8, 0.2, 1)  # Green
            elif abs_dist <= abs(mean + std):
                border_color = (0.8, 0.8, 0.2, 1)  # Yellow
            else:
                border_color = (0.8, 0.2, 0.2, 1)  # Red
            
            border_batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": vertices})
            shader.uniform_float("color", border_color)
            border_batch.draw(shader)
            
            # Draw measurement number and direction - scaled font
            blf.size(0, int(11 * scale))
            blf.color(0, 0.2, 0.2, 0.2, 1)
            blf.position(0, mx - label_width/2 + 5*scale, my + offset_y + label_height - 18*scale, 0)
            dir_short = "S→T" if measurement_dir == 'SOURCE_TO_TARGET' else "T→S"
            blf.draw(0, f"M{i+1} ({dir_short})")
            
            # Draw distance value - scaled font
            blf.size(0, int(14 * scale))
            if distance >= 0:
                blf.color(0, 0.8, 0.1, 0.1, 1)
                sign = "+"
            else:
                blf.color(0, 0.1, 0.1, 0.8, 1)
                sign = ""
            
            blf.position(0, mx - label_width/2 + 5*scale, my + offset_y + 5*scale, 0)
            blf.draw(0, f"{sign}{distance:.3f}")
            
            # Draw marker at measurement point - scaled
            marker_size = 5 * scale
            marker_vertices = [
                (mx - marker_size, my), (mx + marker_size, my),
                (mx, my - marker_size), (mx, my + marker_size)
            ]
            marker_batch = batch_for_shader(shader, 'LINES', {"pos": marker_vertices})
            shader.uniform_float("color", (0.2, 0.2, 0.2, 1))
            marker_batch.draw(shader)
        
        # Draw current probe
        if not probe_data['active']:
            return
        
        x, y = probe_data['position']
        distance = probe_data['distance']
        vertex_idx = probe_data['vertex_index']
        world_pos = probe_data.get('world_pos', Vector((0, 0, 0)))
        
        # Get zoom scale for active probe
        scale = get_view_scale(context, world_pos)
        
        # Scaled box dimensions
        box_width = 280 * scale
        box_height = 140 * scale
        box_offset = 20 * scale
        padding = 10 * scale
        
        box_x = x + box_offset
        box_y = y + box_offset
        
        # Keep box within screen bounds
        if box_x + box_width > context.area.width:
            box_x = x - box_width - box_offset
        if box_y + box_height > context.area.height:
            box_y = y - box_height - box_offset
        
        # Draw background
        vertices = [
            (box_x, box_y),
            (box_x + box_width, box_y),
            (box_x + box_width, box_y + box_height),
            (box_x, box_y + box_height)
        ]
        
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'TRI_FAN', {"pos": vertices})
        shader.bind()
        shader.uniform_float("color", (0.9, 0.9, 0.9, 0.95))
        batch.draw(shader)
        
        # Draw colored border based on deviation
        abs_dist = abs(distance)
        if abs_dist <= abs(mean):
            border_color = (0.2, 0.8, 0.2, 1)
        elif abs_dist <= abs(mean + std):
            border_color = (0.8, 0.8, 0.2, 1)
        else:
            border_color = (0.8, 0.2, 0.2, 1)
        
        # Draw thick border - scaled
        border_width = max(1, int(3 * scale))
        for offset in range(border_width):
            border_vertices = [
                (box_x - offset, box_y - offset),
                (box_x + box_width + offset, box_y - offset),
                (box_x + box_width + offset, box_y + box_height + offset),
                (box_x - offset, box_y + box_height + offset)
            ]
            border_batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": border_vertices})
            shader.uniform_float("color", border_color)
            border_batch.draw(shader)
        
        # Draw text - scaled fonts
        blf.size(0, int(14 * scale))
        blf.color(0, 0.2, 0.2, 0.2, 1)
        
        # Title with direction
        blf.position(0, box_x + padding, box_y + box_height - 25*scale, 0)
        blf.draw(0, f"Deviation Analysis ({dir_text})")
        
        # Deviation value
        blf.size(0, int(20 * scale))
        if distance >= 0:
            blf.color(0, 0.8, 0.0, 0.0, 1)
            sign = "+"
        else:
            blf.color(0, 0.0, 0.0, 0.8, 1)
            sign = ""
        
        blf.position(0, box_x + padding, box_y + box_height - 55*scale, 0)
        blf.draw(0, f"Deviation: {sign}{distance:.4f} mm")
        
        # Additional info
        blf.size(0, int(12 * scale))
        blf.color(0, 0.3, 0.3, 0.3, 1)
        blf.position(0, box_x + padding, box_y + 50*scale, 0)
        blf.draw(0, f"Vertex: {vertex_idx}")
        
        blf.position(0, box_x + padding, box_y + 30*scale, 0)
        if std > 0:
            blf.draw(0, f"σ from mean: {abs((distance - mean) / std):.2f}")
        else:
            blf.draw(0, "σ from mean: N/A")
        
        blf.position(0, box_x + padding, box_y + 15*scale, 0)
        blf.draw(0, f"Mean: {mean:.4f} mm")
        
        # Instruction text
        blf.color(0, 0.2, 0.4, 0.8, 1)
        blf.position(0, box_x + padding, box_y + 5*scale, 0)
        blf.draw(0, "Click to save")
        
        # Draw crosshair at cursor - scaled
        line_length = 15 * scale
        crosshair_vertices = [
            (x - line_length, y), (x + line_length, y),
            (x, y - line_length), (x, y + line_length)
        ]
        
        circle_segments = 16
        circle_radius = 8 * scale
        circle_vertices = []
        for i in range(circle_segments + 1):
            angle = (i / circle_segments) * 2 * np.pi
            circle_vertices.append((
                x + circle_radius * np.cos(angle),
                y + circle_radius * np.sin(angle)
            ))
        
        crosshair_batch = batch_for_shader(shader, 'LINES', {"pos": crosshair_vertices})
        shader.uniform_float("color", (0.2, 0.2, 0.2, 0.8))
        crosshair_batch.draw(shader)
        
        circle_batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": circle_vertices})
        shader.uniform_float("color", (0.3, 0.3, 0.3, 0.6))
        circle_batch.draw(shader)
    
    class MESH_OT_show_legend_distance(Operator):
        """Show color scale legend"""
        bl_idname = "mesh.show_legend_distance"
        bl_label = "Show Legend"
        
        _handle = None
        _is_dragging = False
        _drag_offset_x = 0
        _drag_offset_y = 0
        
        def modal(self, context, event):
            context.area.tag_redraw()
            
            # Check for mouse wheel scaling
            if event.type == 'WHEELUPMOUSE' or event.type == 'WHEELDOWNMOUSE':
                x = legend_position['x'] - 15 * legend_scale['factor']
                y = context.area.height - legend_position['y'] - 60 * legend_scale['factor']
                width = (60 + 110 + 15) * legend_scale['factor']  # Genişlik 60 piksel için güncellendi
                height = (500 + 120) * legend_scale['factor']
                
                if (event.mouse_region_x >= x and event.mouse_region_x <= x + width and
                    event.mouse_region_y >= y - height and event.mouse_region_y <= y):
                    if event.type == 'WHEELUPMOUSE':
                        legend_scale['factor'] = min(3.0, legend_scale['factor'] * 1.1)
                    else:
                        legend_scale['factor'] = max(0.5, legend_scale['factor'] * 0.9)
                    return {'RUNNING_MODAL'}
            
            if event.type == 'LEFTMOUSE':
                x = legend_position['x'] - 15 * legend_scale['factor']
                y = context.area.height - legend_position['y'] - 60 * legend_scale['factor']
                width = (60 + 110 + 15) * legend_scale['factor']  # Genişlik 60 piksel için güncellendi
                height = (500 + 120) * legend_scale['factor']
                
                if (event.mouse_region_x >= x and event.mouse_region_x <= x + width and
                    event.mouse_region_y >= y - height and event.mouse_region_y <= y):
                    if event.value == 'PRESS':
                        self._is_dragging = True
                        self._drag_offset_x = event.mouse_region_x - legend_position['x']
                        self._drag_offset_y = event.mouse_region_y - (context.area.height - legend_position['y'])
                        return {'RUNNING_MODAL'}
                    elif event.value == 'RELEASE':
                        self._is_dragging = False
                        return {'PASS_THROUGH'}
                        
            elif event.type == 'MOUSEMOVE' and self._is_dragging:
                legend_position['x'] = event.mouse_region_x - self._drag_offset_x
                legend_position['y'] = context.area.height - event.mouse_region_y + self._drag_offset_y
                return {'RUNNING_MODAL'}
                
            elif event.type in {'RIGHTMOUSE', 'ESC'}:
                self.cancel(context)
                return {'CANCELLED'}
            
            return {'PASS_THROUGH'}
        
        def execute(self, context):
            if MESH_OT_show_legend_distance._handle is None:
                MESH_OT_show_legend_distance._handle = bpy.types.SpaceView3D.draw_handler_add(
                    draw_color_legend, (context,), 'WINDOW', 'POST_PIXEL')
                context.scene.mesh_distance.show_legend = True
            
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        
        def cancel(self, context):
            if MESH_OT_show_legend_distance._handle is not None:
                bpy.types.SpaceView3D.draw_handler_remove(MESH_OT_show_legend_distance._handle, 'WINDOW')
                MESH_OT_show_legend_distance._handle = None
                context.scene.mesh_distance.show_legend = False
            
            self._is_dragging = False
            
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    
    class MESH_OT_hide_legend_distance(Operator):
        """Hide color scale legend"""
        bl_idname = "mesh.hide_legend_distance"
        bl_label = "Hide Legend"
        
        def execute(self, context):
            if MESH_OT_show_legend_distance._handle is not None:
                bpy.types.SpaceView3D.draw_handler_remove(MESH_OT_show_legend_distance._handle, 'WINDOW')
                MESH_OT_show_legend_distance._handle = None
                context.scene.mesh_distance.show_legend = False
            
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            
            return {'FINISHED'}
    
    class MESH_OT_show_histogram_distance(Operator):
        """Show distance histogram"""
        bl_idname = "mesh.show_histogram_distance"
        bl_label = "Show Histogram"
        
        _handle = None
        _is_dragging = False
        _drag_offset_x = 0
        _drag_offset_y = 0
        
        def modal(self, context, event):
            context.area.tag_redraw()
            
            # Check for mouse wheel scaling
            if event.type == 'WHEELUPMOUSE' or event.type == 'WHEELDOWNMOUSE':
                x = histogram_position['x']
                y = context.area.height - histogram_position['y']
                width = 800 * histogram_scale['factor']
                height = 600 * histogram_scale['factor']
                
                if (event.mouse_region_x >= x and event.mouse_region_x <= x + width and
                    event.mouse_region_y >= y - height and event.mouse_region_y <= y):
                    if event.type == 'WHEELUPMOUSE':
                        histogram_scale['factor'] = min(3.0, histogram_scale['factor'] * 1.1)
                    else:
                        histogram_scale['factor'] = max(0.5, histogram_scale['factor'] * 0.9)
                    return {'RUNNING_MODAL'}
            
            if event.type == 'LEFTMOUSE':
                x = histogram_position['x']
                y = context.area.height - histogram_position['y']
                width = 800 * histogram_scale['factor']
                height = 600 * histogram_scale['factor']
                
                if (event.mouse_region_x >= x and event.mouse_region_x <= x + width and
                    event.mouse_region_y >= y - height and event.mouse_region_y <= y):
                    if event.value == 'PRESS':
                        self._is_dragging = True
                        self._drag_offset_x = event.mouse_region_x - x
                        self._drag_offset_y = event.mouse_region_y - y
                        return {'RUNNING_MODAL'}
                    elif event.value == 'RELEASE':
                        self._is_dragging = False
                        return {'PASS_THROUGH'}
                        
            elif event.type == 'MOUSEMOVE' and self._is_dragging:
                histogram_position['x'] = event.mouse_region_x - self._drag_offset_x
                histogram_position['y'] = context.area.height - event.mouse_region_y + self._drag_offset_y
                return {'RUNNING_MODAL'}
                
            elif event.type in {'RIGHTMOUSE', 'ESC'}:
                self.cancel(context)
                return {'CANCELLED'}
            
            return {'PASS_THROUGH'}
        
        def execute(self, context):
            if MESH_OT_show_histogram_distance._handle is None:
                MESH_OT_show_histogram_distance._handle = bpy.types.SpaceView3D.draw_handler_add(
                    draw_distance_histogram, (context,), 'WINDOW', 'POST_PIXEL')
                context.scene.mesh_distance.show_histogram = True
            
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        
        def cancel(self, context):
            if MESH_OT_show_histogram_distance._handle is not None:
                bpy.types.SpaceView3D.draw_handler_remove(MESH_OT_show_histogram_distance._handle, 'WINDOW')
                MESH_OT_show_histogram_distance._handle = None
                context.scene.mesh_distance.show_histogram = False
            
            self._is_dragging = False
            
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    
    class MESH_OT_hide_histogram_distance(Operator):
        """Hide distance histogram"""
        bl_idname = "mesh.hide_histogram_distance"
        bl_label = "Hide Histogram"
        
        def execute(self, context):
            if MESH_OT_show_histogram_distance._handle is not None:
                bpy.types.SpaceView3D.draw_handler_remove(MESH_OT_show_histogram_distance._handle, 'WINDOW')
                MESH_OT_show_histogram_distance._handle = None
                context.scene.mesh_distance.show_histogram = False
            
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            
            return {'FINISHED'}
    
    class MESH_OT_distance_probe_distance(Operator):
        """Enable distance probe - hover over mesh to see distance values"""
        bl_idname = "mesh.distance_probe_distance"
        bl_label = "Distance Probe"
        bl_options = {'REGISTER'}
        
        _handle = None
        _timer = None
        
        @classmethod
        def poll(cls, context):
            settings = context.scene.mesh_distance
            return settings.distance_calculation_complete
        
        def modal(self, context, event):
            context.area.tag_redraw()
            
            if event.type == 'MOUSEMOVE':
                # Update probe position
                probe_data['position'] = (event.mouse_region_x, event.mouse_region_y)
                
                # Get the source mesh
                source_mesh_name = context.scene.get('source_mesh_name')
                if source_mesh_name:
                    source_obj = bpy.data.objects.get(source_mesh_name)
                    if source_obj:
                        # Ray cast to find vertex under cursor
                        region = context.region
                        rv3d = context.region_data
                        coord = event.mouse_region_x, event.mouse_region_y
                        
                        # Get ray from view
                        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
                        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
                        
                        # Transform ray to object space
                        matrix_inv = source_obj.matrix_world.inverted()
                        ray_origin_obj = matrix_inv @ ray_origin
                        ray_target_obj = matrix_inv @ (ray_origin + view_vector)
                        ray_direction_obj = (ray_target_obj - ray_origin_obj).normalized()
                        
                        # Cast ray
                        success, location, normal, face_index = source_obj.ray_cast(
                            ray_origin_obj, ray_direction_obj)
                        
                        if success:
                            # Find closest vertex
                            mesh = source_obj.data
                            face = mesh.polygons[face_index]
                            
                            # Get vertex positions
                            min_dist = float('inf')
                            closest_vert_idx = -1
                            
                            for vert_idx in face.vertices:
                                vert_pos = mesh.vertices[vert_idx].co
                                dist = (vert_pos - location).length
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_vert_idx = vert_idx
                            
                            # Get distance value
                            distances = context.scene.get('mesh_distances')
                            if distances and closest_vert_idx < len(distances):
                                probe_data['active'] = True
                                probe_data['distance'] = distances[closest_vert_idx]
                                probe_data['vertex_index'] = closest_vert_idx
                                probe_data['world_pos'] = source_obj.matrix_world @ mesh.vertices[closest_vert_idx].co
                            else:
                                probe_data['active'] = False
                        else:
                            probe_data['active'] = False
            
            elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
                # Save current measurement if probe is active
                if probe_data['active']:
                    current_dir = context.scene.get('current_direction', 'SOURCE_TO_TARGET')
                    saved_measurements.append({
                        'screen_pos': (event.mouse_region_x, event.mouse_region_y),
                        'distance': probe_data['distance'],
                        'vertex_index': probe_data['vertex_index'],
                        'world_pos': probe_data['world_pos'].copy(),
                        'direction': current_dir
                    })
                    self.report({'INFO'}, f"Measurement M{len(saved_measurements)} saved: {probe_data['distance']:.4f} mm")
                return {'RUNNING_MODAL'}
            
            elif event.type == 'C' and event.value == 'PRESS':
                # Clear all saved measurements
                saved_measurements.clear()
                self.report({'INFO'}, "All measurements cleared")
                return {'RUNNING_MODAL'}
                    
            elif event.type in {'RIGHTMOUSE', 'ESC'}:
                self.cancel(context)
                return {'CANCELLED'}
            
            return {'PASS_THROUGH'}
        
        def execute(self, context):
            # Add draw handler
            if MESH_OT_distance_probe_distance._handle is None:
                MESH_OT_distance_probe_distance._handle = bpy.types.SpaceView3D.draw_handler_add(
                    draw_distance_probe, (context,), 'WINDOW', 'POST_PIXEL')
                context.scene.mesh_distance.show_probe = True
            
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        
        def cancel(self, context):
            if MESH_OT_distance_probe_distance._handle is not None:
                bpy.types.SpaceView3D.draw_handler_remove(MESH_OT_distance_probe_distance._handle, 'WINDOW')
                MESH_OT_distance_probe_distance._handle = None
                context.scene.mesh_distance.show_probe = False
            
            probe_data['active'] = False
            
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    
    class MESH_OT_disable_probe_distance(Operator):
        """Disable distance probe"""
        bl_idname = "mesh.disable_probe_distance"
        bl_label = "Disable Probe"
        
        def execute(self, context):
            if MESH_OT_distance_probe_distance._handle is not None:
                bpy.types.SpaceView3D.draw_handler_remove(MESH_OT_distance_probe_distance._handle, 'WINDOW')
                MESH_OT_distance_probe_distance._handle = None
                context.scene.mesh_distance.show_probe = False
            
            probe_data['active'] = False
            
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            
            return {'FINISHED'}
    
    class MESH_OT_clear_measurements_distance(Operator):
        """Clear all saved deviation measurements"""
        bl_idname = "mesh.clear_measurements_distance"
        bl_label = "Clear Measurements"
        
        def execute(self, context):
            saved_measurements.clear()
            self.report({'INFO'}, "All measurements cleared")
            
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            
            return {'FINISHED'}

# PANELS

class VIEW3D_PT_mesh_distance_main(Panel):
    """Main panel"""
    bl_label = "Mesh Distance Analysis"
    bl_idname = "VIEW3D_PT_mesh_distance_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Distance Analysis"
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.mesh_distance
        
        # Open3D check
        row = layout.row()
        if settings.open3d_available:
            try:
                import open3d as o3d
                row.operator("mesh.check_open3d_distance", 
                           text=f"Open3D {o3d.__version__} ✓", 
                           icon='CHECKMARK')
            except:
                row.operator("mesh.check_open3d_distance", text="Check Open3D", icon='ERROR')
        else:
            row.operator("mesh.check_open3d_distance", text="Check Open3D", icon='ERROR')
        
        # Mesh selection
        layout.separator()

        # Offer the pair already set up in the Mesh Alignment addon
        test, ref = alignment_meshes(context)
        if test and ref:
            in_sync = (settings.source_mesh == test and settings.target_mesh == ref)
            row = layout.row()
            row.enabled = not in_sync
            row.operator("mesh.use_alignment_meshes",
                         text="Mesh Alignment pair loaded" if in_sync else "Use Mesh Alignment Pair",
                         icon='CHECKMARK' if in_sync else 'IMPORT')

        layout.prop(settings, "source_mesh", icon='MESH_DATA')
        layout.prop(settings, "target_mesh", icon='MESH_DATA')
        
        if settings.source_mesh and settings.target_mesh:
            # Show mesh info
            box = layout.box()
            box.label(text="Mesh Information:", icon='INFO')
            col = box.column(align=True)
            col.scale_y = 0.8
            col.label(text=f"Source: {len(settings.source_mesh.data.vertices):,} vertices")
            col.label(text=f"Target: {len(settings.target_mesh.data.vertices):,} vertices")

class VIEW3D_PT_surface_sampling(Panel):
    """Surface sampling settings panel"""
    bl_label = "Hybrid Sampling Mode"
    bl_idname = "VIEW3D_PT_surface_sampling"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Distance Analysis"
    bl_parent_id = "VIEW3D_PT_mesh_distance_main"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw_header(self, context):
        self.layout.prop(context.scene.mesh_distance, "use_surface_sampling", text="")
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.mesh_distance
        layout.enabled = settings.use_surface_sampling
        
        # Info box
        box = layout.box()
        box.label(text="Hybrid Mode:", icon='INFO')
        col = box.column(align=True)
        col.scale_y = 0.8
        col.label(text="• Stats: Point-based (accurate)")
        col.label(text="• Visual: Vertex-based (sharp)")
        
        # Sampling method
        layout.prop(settings, "sampling_method")
        
        # Density settings
        box = layout.box()
        box.label(text="Sampling Density:", icon='OUTLINER_OB_POINTCLOUD')
        col = box.column(align=True)
        col.prop(settings, "sample_density", text="Points/mm²")
        col.prop(settings, "max_sample_points", text="Max Points")
        
        # Info about last calculation
        if settings.distance_calculation_complete and settings.use_surface_sampling:
            box = layout.box()
            box.label(text="Last Calculation:", icon='CHECKMARK')
            col = box.column(align=True)
            col.scale_y = 0.8
            if settings.actual_sample_count_st > 0:
                col.label(text=f"S→T: {settings.actual_sample_count_st:,} samples")
            if settings.actual_sample_count_ts > 0:
                col.label(text=f"T→S: {settings.actual_sample_count_ts:,} samples")

class VIEW3D_PT_distance_calculation(Panel):
    """Distance calculation panel"""
    bl_label = "Distance Calculation"
    bl_idname = "VIEW3D_PT_distance_calculation"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Distance Analysis"
    bl_parent_id = "VIEW3D_PT_mesh_distance_main"
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.mesh_distance
        
        # Direction Switch Buttons (always show)
        if settings.distance_calculation_complete:
            box = layout.box()
            box.label(text="Active Direction:", icon='ARROW_LEFTRIGHT')
            row = box.row(align=True)
            op = row.operator("mesh.switch_direction", text="Source → Target", icon='FORWARD')
            op.direction = 'SOURCE_TO_TARGET'
            row.enabled = (settings.active_direction != 'SOURCE_TO_TARGET')
            
            row2 = box.row(align=True)
            op = row2.operator("mesh.switch_direction", text="Target → Source", icon='BACK')
            op.direction = 'TARGET_TO_SOURCE'
            row2.enabled = (settings.active_direction != 'TARGET_TO_SOURCE')
        
        # Settings
        box = layout.box()
        box.label(text="Settings:", icon='SETTINGS')
        box.prop(settings, "use_signed_distance")
        
        # Calculate button
        row = layout.row()
        row.scale_y = 1.5
        row.operator("mesh.calculate_mesh_distances", icon='PLAY')
        has_pair = bool(settings.source_mesh and settings.target_mesh)
        if not has_pair:
            _t, _r = alignment_meshes(context)
            has_pair = bool(_t and _r)
        row.enabled = bool(has_pair and settings.open3d_available)
        
        # Results
        if settings.distance_calculation_complete:
            box = layout.box()
            box.label(text="Distance Results:", icon='INFO')
            
            # Show both direction results
            col = box.column(align=True)
            col.label(text="Source → Target:", icon='FORWARD')
            row = col.row()
            row.label(text="Mean (signed):")
            row.label(text=f"{settings.mean_signed_st:+.4f} mm")
            row = col.row()
            row.label(text="Std Dev:")
            row.label(text=f"{settings.std_deviation_st:.4f} mm")
            row = col.row()
            row.label(text="MAD:")
            row.label(text=f"{settings.mad_distance_st:.4f} mm")
            row = col.row()
            row.label(text="RMS:")
            row.label(text=f"{settings.rms_distance_st:.4f} mm")
            row = col.row()
            row.label(text="95th Perc.:")
            row.label(text=f"{settings.p95_distance_st:.4f} mm")
            row = col.row()
            row.label(text="Max:")
            row.label(text=f"{settings.max_distance_st:.4f} mm")
            
            col.separator()
            col.label(text="Target → Source:", icon='BACK')
            row = col.row()
            row.label(text="Mean (signed):")
            row.label(text=f"{settings.mean_signed_ts:+.4f} mm")
            row = col.row()
            row.label(text="Std Dev:")
            row.label(text=f"{settings.std_deviation_ts:.4f} mm")
            row = col.row()
            row.label(text="MAD:")
            row.label(text=f"{settings.mad_distance_ts:.4f} mm")
            row = col.row()
            row.label(text="RMS:")
            row.label(text=f"{settings.rms_distance_ts:.4f} mm")
            row = col.row()
            row.label(text="95th Perc.:")
            row.label(text=f"{settings.p95_distance_ts:.4f} mm")
            row = col.row()
            row.label(text="Max:")
            row.label(text=f"{settings.max_distance_ts:.4f} mm")
            
            col.separator()
            col.label(text="Combined:", icon='ARROW_LEFTRIGHT')
            row = col.row()
            row.label(text="Hausdorff:")
            row.label(text=f"{settings.hausdorff_distance:.4f} mm")
            row = col.row()
            row.label(text="95th Perc. (max):")
            row.label(text=f"{settings.hausdorff_percentile_95:.4f} mm")

class VIEW3D_PT_deviation_analysis(Panel):
    """Deviation analysis panel"""
    bl_label = "Deviation Analysis"
    bl_idname = "VIEW3D_PT_deviation_analysis"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Distance Analysis"
    bl_parent_id = "VIEW3D_PT_mesh_distance_main"
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.mesh_distance
        
        if not settings.distance_calculation_complete:
            layout.label(text="Calculate distances first", icon='ERROR')
            return
        
        # Probe button (main feature)
        col = layout.column()
        col.scale_y = 1.5
        if settings.show_probe:
            col.operator("mesh.disable_probe_distance", text="Disable Deviation Probe", icon='RESTRICT_VIEW_ON')
        else:
            col.operator("mesh.distance_probe_distance", text="Enable Deviation Probe", icon='VIEW_CAMERA')
        
        if settings.show_probe:
            row = col.row()
            row.operator("mesh.clear_measurements_distance", text="Clear Measurements", icon='X')
            row.scale_y = 0.8
            
            # Help text
            box = layout.box()
            box.scale_y = 0.8
            box.label(text="• Hover over mesh to see values", icon='INFO')
            box.label(text="• Left-click to save measurement")
            box.label(text="• Press 'C' to clear all")
            box.label(text="• Right-click/ESC to exit")

class VIEW3D_PT_color_mapping(Panel):
    """Color mapping panel"""
    bl_label = "Color Mapping"
    bl_idname = "VIEW3D_PT_color_mapping"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Distance Analysis"
    bl_parent_id = "VIEW3D_PT_mesh_distance_main"
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.mesh_distance
        
        if not settings.distance_calculation_complete:
            layout.label(text="Calculate distances first", icon='ERROR')
            return
        
        # Color mapping settings
        box = layout.box()
        box.label(text="Range Settings:", icon='COLOR')
        
        box.prop(settings, "auto_range")
        
        col = box.column(align=True)
        col.enabled = not settings.auto_range
        
        row = col.row(align=True)
        row.label(text="Min:")
        row.prop(settings, "range_min", text="")
        
        row = col.row(align=True)
        row.label(text="Max:")
        row.prop(settings, "range_max", text="")
        
        # Outlier settings
        box.separator()
        box.prop(settings, "highlight_outliers")
        if settings.highlight_outliers:
            box.prop(settings, "outlier_color")
        
        # Update button
        if not settings.auto_range or settings.highlight_outliers:
            layout.operator("mesh.apply_color_mapping_distance", text="Update Colors", icon='FILE_REFRESH')

class VIEW3D_PT_smoothing_filters(Panel):
    """Smoothing filters panel"""
    bl_label = "Smoothing Filters"
    bl_idname = "VIEW3D_PT_smoothing_filters"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Distance Analysis"
    bl_parent_id = "VIEW3D_PT_mesh_distance_main"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.mesh_distance
        
        if not settings.distance_calculation_complete:
            layout.label(text="Calculate distances first", icon='ERROR')
            return
        
        # Gaussian filter
        col = layout.column(align=True)
        col.prop(settings, "use_gaussian_filter")
        if settings.use_gaussian_filter:
            row = col.row(align=True)
            row.prop(settings, "gaussian_sigma")
            row.prop(settings, "gaussian_iterations")
        
        # Laplacian smoothing
        col.separator()
        col.prop(settings, "use_laplacian_smooth")
        if settings.use_laplacian_smooth:
            row = col.row(align=True)
            row.prop(settings, "laplacian_factor", slider=True)
            row.prop(settings, "laplacian_iterations")
        
        # Update button
        if settings.use_gaussian_filter or settings.use_laplacian_smooth:
            layout.separator()
            layout.operator("mesh.apply_color_mapping_distance", text="Apply Filters", icon='FILE_REFRESH')

class VIEW3D_PT_visualization_options(Panel):
    """Visualization options panel"""
    bl_label = "Visualization Options"
    bl_idname = "VIEW3D_PT_visualization_options"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Distance Analysis"
    bl_parent_id = "VIEW3D_PT_mesh_distance_main"
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.mesh_distance
        
        if not settings.distance_calculation_complete:
            layout.label(text="Calculate distances first", icon='ERROR')
            return
        
        if not GPU_AVAILABLE:
            layout.label(text="GPU module not available", icon='ERROR')
            return
        
        # Color Scale Legend
        box = layout.box()
        box.label(text="Color Scale:", icon='COLOR')
        row = box.row()
        if settings.show_legend:
            row.operator("mesh.hide_legend_distance", text="Hide Legend", icon='HIDE_ON')
        else:
            row.operator("mesh.show_legend_distance", text="Show Legend", icon='HIDE_OFF')
        
        # Histogram
        box = layout.box()
        box.label(text="Histogram:", icon='GRAPH')
        row = box.row()
        if settings.show_histogram:
            row.operator("mesh.hide_histogram_distance", text="Hide Histogram", icon='HIDE_ON')
        else:
            row.operator("mesh.show_histogram_distance", text="Show Histogram", icon='HIDE_OFF')
        
        # Help text
        if settings.show_legend or settings.show_histogram:
            help_box = layout.box()
            help_box.scale_y = 0.8
            help_box.label(text="• Drag to move", icon='INFO')
            help_box.label(text="• Scroll wheel to scale")
            help_box.label(text="• Right-click/ESC to close")

# Registration
classes = [
    MeshDistanceSettings,
    MESH_OT_check_open3d_distance,
    MESH_OT_use_alignment_meshes,
    MESH_OT_calculate_mesh_distances,
    MESH_OT_switch_direction,
    MESH_OT_visualize_distances,
    MESH_OT_apply_color_mapping_distance,
    VIEW3D_PT_mesh_distance_main,
    VIEW3D_PT_surface_sampling,
    VIEW3D_PT_distance_calculation,
    VIEW3D_PT_deviation_analysis,
    VIEW3D_PT_color_mapping,
    VIEW3D_PT_smoothing_filters,
    VIEW3D_PT_visualization_options,
]

if GPU_AVAILABLE:
    classes.extend([
        MESH_OT_show_legend_distance,
        MESH_OT_hide_legend_distance,
        MESH_OT_show_histogram_distance,
        MESH_OT_hide_histogram_distance,
        MESH_OT_distance_probe_distance,
        MESH_OT_disable_probe_distance,
        MESH_OT_clear_measurements_distance,
    ])

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.mesh_distance = PointerProperty(type=MeshDistanceSettings)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Scene.mesh_distance

if __name__ == "__main__":
    register()