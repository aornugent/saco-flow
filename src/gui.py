"""Real-time 3D visualization system using Taichi UI.

Implements decoupled resolution rendering and layered material system
as specified in docs/visualisation.md.
"""

import math

import taichi as ti


@ti.data_oriented
class Visualizer3D:
    """3D Visualizer for Saco-Flow simulation.

    Handles rendering of terrain, water, and vegetation using a dedicated
    visualization mesh that is decoupled from the physics simulation grid.
    """

    def __init__(self, vis_n: int = 512, window_title: str = "Saco-Flow", headless: bool = False):
        """Initialize visualizer.

        Args:
            vis_n: Resolution of the visualization mesh (MxM).
            window_title: Title of the window.
            headless: If True, do not create window (for testing).
        """
        self.vis_n = vis_n
        self.headless = headless
        
        # Rendering fields (MxM vertices)
        self.num_verts = vis_n * vis_n
        self.num_indices = (vis_n - 1) * (vis_n - 1) * 6
        
        self.vis_verts = ti.Vector.field(3, dtype=float, shape=self.num_verts)
        self.vis_colors = ti.Vector.field(3, dtype=float, shape=self.num_verts)
        self.vis_indices = ti.field(dtype=int, shape=self.num_indices)
        
        # Initialize UI
        if not self.headless:
            self.window = ti.ui.Window(window_title, (1280, 720), vsync=True)
            self.canvas = self.window.get_canvas()
            self.scene = self.window.get_scene()
            self.camera = ti.ui.Camera()
            
            # Camera defaults
            self.camera.position(0.5, 1.2, 1.2)
            self.camera.lookat(0.5, 0.0, 0.5)
            self.camera.up(0.0, 1.0, 0.0)
        else:
            self.window = None
            self.canvas = None
            self.scene = None
            self.camera = None
        
        # Initialize mesh topology once
        self.init_mesh_indices()

    @ti.kernel
    def init_mesh_indices(self):
        """Generate triangle indices for the grid mesh."""
        for i, j in ti.ndrange(self.vis_n - 1, self.vis_n - 1):
            quad_idx = i * (self.vis_n - 1) + j
            
            # Vertices of the quad
            v00 = i * self.vis_n + j
            v01 = i * self.vis_n + (j + 1)
            v10 = (i + 1) * self.vis_n + j
            v11 = (i + 1) * self.vis_n + (j + 1)
            
            # Triangle 1
            self.vis_indices[quad_idx * 6 + 0] = v00
            self.vis_indices[quad_idx * 6 + 1] = v01
            self.vis_indices[quad_idx * 6 + 2] = v11
            
            # Triangle 2
            self.vis_indices[quad_idx * 6 + 3] = v00
            self.vis_indices[quad_idx * 6 + 4] = v11
            self.vis_indices[quad_idx * 6 + 5] = v10

    @ti.kernel
    def update_mesh(
        self,
        Z: ti.template(),
        h: ti.template(),
        M: ti.template(),
        P: ti.template(),
        M_sat: float,
    ):
        """Sample simulation fields and update visualization mesh.
        
        Args:
            Z: Terrain elevation field (NxN)
            h: Surface water depth field (NxN)
            M: Soil moisture field (NxN)
            P: Vegetation biomass field (NxN)
            M_sat: Soil saturation capacity
        """
        sim_n = Z.shape[0]  # Assuming square physics grid
        
        for i, j in ti.ndrange(self.vis_n, self.vis_n):
            idx = i * self.vis_n + j
            
            # Normalized coordinates [0, 1]
            u = i / (self.vis_n - 1)
            v = j / (self.vis_n - 1)
            
            # Nearest neighbor sampling from physics grid
            # (Could use bilinear interpolation for smoother results, but NN is faster/simpler)
            si = int(u * (sim_n - 1))
            sj = int(v * (sim_n - 1))
            
            z_val = Z[si, sj]
            h_val = h[si, sj]
            m_val = M[si, sj]
            p_val = P[si, sj]
            
            # --- Geometry ---
            # S_vert = 2.0
            y_vis = z_val * 2.0
            
            # Water depth offset logic
            delta_water = 0.0
            if h_val >= 1e-3:  # 1mm threshold
                delta_water = 0.1 + 0.5 * ti.min(h_val, 1.0)
                
            y_vis += delta_water
            
            # Coordinate system: x=u, y=elevation, z=v (mapped to 0-1 range roughly)
            self.vis_verts[idx] = ti.Vector([u, y_vis, v])
            
            # --- Materiality (Color) ---
            
            # 1. Soil Layer
            saturation = ti.min(m_val / M_sat, 1.0)
            
            color_dry = ti.Vector([0.80, 0.75, 0.65])
            color_wet = ti.Vector([0.35, 0.25, 0.15])
            
            soil_color = color_dry * (1.0 - saturation) + color_wet * saturation
            
            # 2. Vegetation Layer
            # Interpolate Sparse <-> Dense
            color_sparse = ti.Vector([0.60, 0.70, 0.20])
            color_dense = ti.Vector([0.05, 0.40, 0.05])
            
            # Visual opacity/density mapping from biomass P
            # Assuming P ranges roughly 0 to 1+
            veg_factor = ti.min(p_val, 1.0)  
            veg_color = color_sparse * (1.0 - veg_factor) + color_dense * veg_factor
            
            # Blend Soil and Veg
            # Basic alpha blending: result = veg * alpha + soil * (1 - alpha)
            # Using biomass as alpha proxy
            veg_alpha = ti.min(p_val * 0.8, 1.0) # Scale up a bit so moderately dense veg covers soil
            
            base_color = veg_color * veg_alpha + soil_color * (1.0 - veg_alpha)
            
            # 3. Water Layer
            color_water = ti.Vector([0.10, 0.40, 0.90])
            
            # Water opacity scales with depth
            water_alpha = 0.0
            if h_val > 1e-4:
                # deeper water -> more opaque. Saturate at 10cm depth
                water_alpha = 0.3 + 0.7 * ti.min(h_val / 0.1, 1.0)
                
            final_color = color_water * water_alpha + base_color * (1.0 - water_alpha)
            
            self.vis_colors[idx] = final_color

    def update(self, sim_state, params):
        """Update visualization from simulation state.
        
        Args:
            sim_state: Current SimulationState object.
            params: SimulationParams object (needed for M_sat).
        """
        self.update_mesh(
            sim_state.fields.Z,
            sim_state.fields.h,
            sim_state.fields.M,
            sim_state.fields.P,
            params.M_sat
        ) 

    def render(self):
        """Render the current frame."""
        if self.headless:
            return

        if self.window.is_pressed(ti.ui.ESCAPE):
            self.window.destroy()
            return

        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)
        
        self.scene.ambient_light((0.5, 0.5, 0.5))
        self.scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))

        self.scene.mesh(self.vis_verts, self.vis_indices, per_vertex_color=self.vis_colors)
        self.canvas.scene(self.scene)
        self.window.show()

