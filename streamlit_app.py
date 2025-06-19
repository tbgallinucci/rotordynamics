import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy.linalg import solve
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Rotordynamic Analysis Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import the analysis class
from main_rotordynamic import RotordynamicAnalysis

class StreamlitRotordynamicApp:
    def __init__(self):
        self.analysis = None
        
    def create_sidebar(self):
        """Create sidebar with parameter controls"""
        
        st.sidebar.title("System Parameters")
        st.sidebar.markdown("---")
        
        # Material Properties
        st.sidebar.subheader("Material Properties")
        E = st.sidebar.number_input(
            "Young's Modulus (GPa)", 
            value=207.0, 
            min_value=100.0, 
            max_value=400.0,
            step=1.0,
            help="Material stiffness property"
        ) * 1e9
        
        rho = st.sidebar.number_input(
            "Density (kg/m³)", 
            value=7850, 
            min_value=5000, 
            max_value=12000,
            step=10,
            help="Material density"
        )
        
        # Shaft Geometry
        st.sidebar.subheader("Shaft Geometry")
        de = st.sidebar.number_input(
            "Shaft Diameter (mm)", 
            value=13.0, 
            min_value=5.0, 
            max_value=50.0,
            step=0.1,
            help="Main shaft diameter"
        ) * 1e-3
        
        le = st.sidebar.number_input(
            "Shaft Length (mm)", 
            value=747.0, 
            min_value=200.0, 
            max_value=2000.0,
            step=1.0,
            help="Total shaft length"
        ) * 1e-3
        
        # Disk Properties
        st.sidebar.subheader("Disk Properties")
        dd = st.sidebar.number_input(
            "Disk Diameter (mm)", 
            value=90.0, 
            min_value=30.0, 
            max_value=200.0,
            step=1.0,
            help="Disk outer diameter"
        ) * 1e-3
        
        md = st.sidebar.number_input(
            "Disk Mass (kg)", 
            value=2.3, 
            min_value=0.5, 
            max_value=10.0,
            step=0.1,
            help="Total disk mass"
        )
        
        me = st.sidebar.number_input(
            "Unbalance (kg·mm)", 
            value=0.037, 
            min_value=0.001, 
            max_value=1.0,
            step=0.001,
            format="%.3f",
            help="Unbalance mass times radius"
        ) * 1e-3
        
        # Bearing Properties
        st.sidebar.subheader("Bearing Properties")
        
        # Bearing type selection
        bearing1_type = st.sidebar.selectbox(
            "Bearing 1 Type",
            ["journal", "ball"],
            index=0,
            help="Select bearing type for bearing 1"
        )
        
        bearing2_type = st.sidebar.selectbox(
            "Bearing 2 Type", 
            ["journal", "ball"],
            index=0,
            help="Select bearing type for bearing 2"
        )
        
        # Journal bearing properties (shown if any bearing is journal type)
        if bearing1_type == "journal" or bearing2_type == "journal":
            st.sidebar.markdown("**Journal Bearing Properties:**")
            D = st.sidebar.number_input(
                "Bearing Diameter (mm)", 
                value=30.0, 
                min_value=10.0, 
                max_value=100.0,
                step=1.0,
                help="Journal bearing diameter"
            ) * 1e-3
            
            C = st.sidebar.number_input(
                "Bearing Length (mm)", 
                value=20.0, 
                min_value=5.0, 
                max_value=100.0,
                step=1.0,
                help="Bearing axial length"
            ) * 1e-3
            
            delta = st.sidebar.number_input(
                "Radial Clearance (μm)", 
                value=90.0, 
                min_value=10.0, 
                max_value=500.0,
                step=1.0,
                help="Bearing radial clearance"
            ) * 1e-6
            
            mi = st.sidebar.number_input(
                "Oil Viscosity (Pa·s)", 
                value=0.051, 
                min_value=0.001, 
                max_value=1.0,
                step=0.001,
                format="%.3f",
                help="Lubricant dynamic viscosity"
            )
        else:
            # Default values for journal bearings (even if not used)
            D = 30e-3
            C = 20e-3
            delta = 90e-6
            mi = 0.051
        
        # Ball bearing properties (shown if any bearing is ball type)
        if bearing1_type == "ball" or bearing2_type == "ball":
            st.sidebar.markdown("**Ball Bearing Properties:**")
            
            if bearing1_type == "ball":
                st.sidebar.markdown("*Bearing 1 (Ball):*")
                ball1_kxx = st.sidebar.number_input(
                    "Bearing 1 Stiffness Kxx (MN/m)",
                    value=100.0,
                    min_value=1.0,
                    max_value=1000.0,
                    step=1.0,
                    help="Ball bearing 1 stiffness in X direction"
                ) * 1e6
                
                ball1_kyy = st.sidebar.number_input(
                    "Bearing 1 Stiffness Kyy (MN/m)",
                    value=100.0,
                    min_value=1.0,
                    max_value=1000.0,
                    step=1.0,
                    help="Ball bearing 1 stiffness in Y direction"
                ) * 1e6
                
                ball1_cxx = st.sidebar.number_input(
                    "Bearing 1 Damping Cxx (kN·s/m)",
                    value=1.0,
                    min_value=0.1,
                    max_value=10.0,
                    step=0.1,
                    help="Ball bearing 1 damping in X direction"
                ) * 1e3
                
                ball1_cyy = st.sidebar.number_input(
                    "Bearing 1 Damping Cyy (kN·s/m)",
                    value=1.0,
                    min_value=0.1,
                    max_value=10.0,
                    step=0.1,
                    help="Ball bearing 1 damping in Y direction"
                ) * 1e3
            else:
                ball1_kxx = ball1_kyy = 1e8
                ball1_cxx = ball1_cyy = 1e3
            
            if bearing2_type == "ball":
                st.sidebar.markdown("*Bearing 2 (Ball):*")
                ball2_kxx = st.sidebar.number_input(
                    "Bearing 2 Stiffness Kxx (MN/m)",
                    value=100.0,
                    min_value=1.0,
                    max_value=1000.0,
                    step=1.0,
                    help="Ball bearing 2 stiffness in X direction"
                ) * 1e6
                
                ball2_kyy = st.sidebar.number_input(
                    "Bearing 2 Stiffness Kyy (MN/m)",
                    value=100.0,
                    min_value=1.0,
                    max_value=1000.0,
                    step=1.0,
                    help="Ball bearing 2 stiffness in Y direction"
                ) * 1e6
                
                ball2_cxx = st.sidebar.number_input(
                    "Bearing 2 Damping Cxx (kN·s/m)",
                    value=1.0,
                    min_value=0.1,
                    max_value=10.0,
                    step=0.1,
                    help="Ball bearing 2 damping in X direction"
                ) * 1e3
                
                ball2_cyy = st.sidebar.number_input(
                    "Bearing 2 Damping Cyy (kN·s/m)",
                    value=1.0,
                    min_value=0.1,
                    max_value=10.0,
                    step=0.1,
                    help="Ball bearing 2 damping in Y direction"
                ) * 1e3
            else:
                ball2_kxx = ball2_kyy = 1e8
                ball2_cxx = ball2_cyy = 1e3
        else:
            # Default values for ball bearings (even if not used)
            ball1_kxx = ball1_kyy = 1e8
            ball1_cxx = ball1_cyy = 1e3
            ball2_kxx = ball2_kyy = 1e8
            ball2_cxx = ball2_cyy = 1e3
        
        # Component Positions
        st.sidebar.subheader("Component Positions")
        bearing1_pos = st.sidebar.number_input(
            "Bearing 1 Position (mm)", 
            value=26.0, 
            min_value=0.0, 
            max_value=le*1000,
            step=1.0,
            help="Distance from shaft start to bearing 1 center"
        ) * 1e-3
        
        disk_pos = st.sidebar.number_input(
            "Disk Position (mm)", 
            value=145.0, 
            min_value=bearing1_pos*1000 + 10, 
            max_value=le*1000 - 10,
            step=1.0,
            help="Distance from shaft start to disk center"
        ) * 1e-3
        
        bearing2_pos = st.sidebar.number_input(
            "Bearing 2 Position (mm)", 
            value=721.0, 
            min_value=disk_pos*1000 + 10, 
            max_value=le*1000,
            step=1.0,
            help="Distance from shaft start to bearing 2 center"
        ) * 1e-3
        
        # Analysis Settings
        st.sidebar.subheader("Analysis Settings")
        omega_min = st.sidebar.number_input(
            "Min Speed (rad/s)", 
            value=10, 
            min_value=1, 
            max_value=100,
            step=1,
            help="Minimum analysis speed"
        )
        
        omega_max = st.sidebar.number_input(
            "Max Speed (rad/s)", 
            value=2000, 
            min_value=500, 
            max_value=5000,
            step=10,
            help="Maximum analysis speed"
        )
        
        omega_step = st.sidebar.number_input(
            "Speed Step (rad/s)", 
            value=5, 
            min_value=1, 
            max_value=50,
            step=1,
            help="Speed increment"
        )
        
        return {
            'E': E, 'rho': rho, 'de': de, 'le': le, 'dd': dd, 'md': md, 'me': me,
            'D': D, 'C': C, 'delta': delta, 'mi': mi,
            'bearing1_type': bearing1_type, 'bearing2_type': bearing2_type,
            'ball1_kxx': ball1_kxx, 'ball1_kyy': ball1_kyy, 'ball1_cxx': ball1_cxx, 'ball1_cyy': ball1_cyy,
            'ball2_kxx': ball2_kxx, 'ball2_kyy': ball2_kyy, 'ball2_cxx': ball2_cxx, 'ball2_cyy': ball2_cyy,
            'bearing1_pos': bearing1_pos, 'disk_pos': disk_pos, 'bearing2_pos': bearing2_pos,
            'omega_min': omega_min, 'omega_max': omega_max, 'omega_step': omega_step
        }
    
    def update_analysis_parameters(self, params):
        """Update analysis object with new parameters"""
        
        if self.analysis is None:
            self.analysis = RotordynamicAnalysis()
        
        # Update material properties
        self.analysis.E = params['E']
        self.analysis.rho = params['rho']
        
        # Update shaft geometry
        self.analysis.de = params['de']
        self.analysis.Ae = (np.pi * params['de']**2) / 4
        self.analysis.le = params['le']
        self.analysis.Ie = (np.pi * params['de']**4) / 64
        
        # Update disk properties
        self.analysis.dd = params['dd']
        self.analysis.md = params['md']
        self.analysis.W = params['md'] * self.analysis.g
        self.analysis.me = params['me']
        self.analysis.e = params['me'] / params['md']
        self.analysis.Ip = (params['md'] * params['dd']**2) / 8
        self.analysis.Id = (params['md'] * params['dd']**2) / 16 + (params['md'] * self.analysis.ld**2) / 12
        
        # Update bearing properties
        self.analysis.D = params['D']
        self.analysis.C = params['C']
        self.analysis.delta = params['delta']
        self.analysis.mi = params['mi']
        
        # Update bearing types
        self.analysis.bearing1_type = params['bearing1_type']
        self.analysis.bearing2_type = params['bearing2_type']
        
        # Update ball bearing properties
        self.analysis.ball_bearing1_kxx = params['ball1_kxx']
        self.analysis.ball_bearing1_kyy = params['ball1_kyy']
        self.analysis.ball_bearing1_cxx = params['ball1_cxx']
        self.analysis.ball_bearing1_cyy = params['ball1_cyy']
        self.analysis.ball_bearing2_kxx = params['ball2_kxx']
        self.analysis.ball_bearing2_kyy = params['ball2_kyy']
        self.analysis.ball_bearing2_cxx = params['ball2_cxx']
        self.analysis.ball_bearing2_cyy = params['ball2_cyy']
        
        # Update component positions for force calculations
        self.analysis.d1 = params['disk_pos'] - params['bearing1_pos']  # Distance from bearing 1 to disk
        self.analysis.d2 = params['bearing2_pos'] - params['bearing1_pos']  # Distance between bearings
        
        # Store positions for visualization (not in analysis object)
        self.bearing1_pos = params['bearing1_pos']
        self.disk_pos = params['disk_pos'] 
        self.bearing2_pos = params['bearing2_pos']
        
        # Update speed range
        self.analysis.omega = np.arange(params['omega_min'], params['omega_max'] + 1, params['omega_step'])
        self.analysis.n = len(self.analysis.omega)
        
        # Recalculate bearing reactions based on new positions
        self.analysis.FM2 = self.analysis.W * self.analysis.d1 / self.analysis.d2
        self.analysis.FM1 = self.analysis.W - self.analysis.FM2
        
        # Reinitialize arrays and FEM model
        self.analysis.initialize_arrays()
        self.analysis.setup_fem_model()
    
    def run_analysis_button(self):
        """Create run analysis button and handle execution"""
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("Run Rotordynamic Analysis", type="primary", use_container_width=True):
                
                if self.analysis is None:
                    st.error("Analysis object not initialized. Please check parameters.")
                    return
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Calculate bearing coefficients
                    status_text.text(f"Calculating bearing 1 coefficients ({self.analysis.bearing1_type})...")
                    progress_bar.progress(20)
                    
                    if self.analysis.bearing1_type == "ball":
                        bearing1_coeff = self.analysis.calculate_ball_bearing_coefficients(1)
                    else:
                        bearing1_coeff = self.analysis.calculate_bearing_coefficients(1, self.analysis.FM1)
                    
                    status_text.text(f"Calculating bearing 2 coefficients ({self.analysis.bearing2_type})...")
                    progress_bar.progress(40)
                    
                    if self.analysis.bearing2_type == "ball":
                        bearing2_coeff = self.analysis.calculate_ball_bearing_coefficients(2)
                    else:
                        bearing2_coeff = self.analysis.calculate_bearing_coefficients(2, self.analysis.FM2)
                    
                    # Store coefficients
                    status_text.text("Processing bearing coefficients...")
                    progress_bar.progress(60)
                    
                    for i in range(self.analysis.n):
                        # Bearing 1
                        self.analysis.epsilon1[i] = bearing1_coeff[f'epsilon_{i}']
                        self.analysis.phi1[i] = bearing1_coeff[f'phi_{i}'] * 180 / np.pi
                        self.analysis.k_m1_11[i] = bearing1_coeff[f'k_11_{i}']
                        self.analysis.k_m1_12[i] = bearing1_coeff[f'k_12_{i}']
                        self.analysis.k_m1_21[i] = bearing1_coeff[f'k_21_{i}']
                        self.analysis.k_m1_22[i] = bearing1_coeff[f'k_22_{i}']
                        self.analysis.d_m1_11[i] = bearing1_coeff[f'd_11_{i}']
                        self.analysis.d_m1_12[i] = bearing1_coeff[f'd_12_{i}']
                        self.analysis.d_m1_21[i] = bearing1_coeff[f'd_21_{i}']
                        self.analysis.d_m1_22[i] = bearing1_coeff[f'd_22_{i}']
                        
                        # Bearing 2
                        self.analysis.epsilon2[i] = bearing2_coeff[f'epsilon_{i}']
                        self.analysis.phi2[i] = bearing2_coeff[f'phi_{i}'] * 180 / np.pi
                        self.analysis.k_m2_11[i] = bearing2_coeff[f'k_11_{i}']
                        self.analysis.k_m2_12[i] = bearing2_coeff[f'k_12_{i}']
                        self.analysis.k_m2_21[i] = bearing2_coeff[f'k_21_{i}']
                        self.analysis.k_m2_22[i] = bearing2_coeff[f'k_22_{i}']
                        self.analysis.d_m2_11[i] = bearing2_coeff[f'd_11_{i}']
                        self.analysis.d_m2_12[i] = bearing2_coeff[f'd_12_{i}']
                        self.analysis.d_m2_21[i] = bearing2_coeff[f'd_21_{i}']
                        self.analysis.d_m2_22[i] = bearing2_coeff[f'd_22_{i}']
                    
                    # Step 2: Calculate critical speeds
                    status_text.text("Calculating critical speeds...")
                    progress_bar.progress(70)
                    
                    self.analysis.calculate_critical_speeds()
                    
                    # Step 3: Calculate system response
                    status_text.text("Calculating frequency response...")
                    progress_bar.progress(85)
                    
                    self.analysis.calculate_system_response()
                    
                    status_text.text("Analysis completed successfully!")
                    progress_bar.progress(100)
                    
                    st.success("✅ Analysis completed successfully!")
                    st.session_state['analysis_complete'] = True
                    st.session_state['analysis_object'] = self.analysis
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.session_state['analysis_complete'] = False
                
                finally:
                    status_text.empty()
                    progress_bar.empty()
    
    def display_results(self):
        """Display analysis results and plots"""
        
        if not st.session_state.get('analysis_complete', False):
            st.info("Please run the analysis first to see results.")
            return
        
        analysis = st.session_state['analysis_object']
        
        # Results summary
        st.subheader("Analysis Results")
        
        # Convert speed range to Hz
        speed_min_hz = analysis.omega[0] / (2 * np.pi)
        speed_max_hz = analysis.omega[-1] / (2 * np.pi)
        
        # Get calculated critical speeds in Hz
        if hasattr(analysis, 'critical_speeds') and len(analysis.critical_speeds) > 0:
            critical_speeds_hz = analysis.critical_speeds / (2 * np.pi)
        else:
            # Fallback to default values if calculation failed
            critical_speeds_hz = [160/(2*np.pi), 750/(2*np.pi), 1460/(2*np.pi)]
        
        # Create columns based on number of critical speeds
        num_critical = len(critical_speeds_hz)
        columns = st.columns(min(4, num_critical + 1))
        
        with columns[0]:
            st.metric("Speed Range", f"{speed_min_hz:.0f} - {speed_max_hz:.0f} Hz")
        
        for i in range(min(3, num_critical)):
            with columns[i + 1]:
                ordinal = ["1st", "2nd", "3rd"][i]
                st.metric(f"{ordinal} Critical Speed", f"{critical_speeds_hz[i]:.1f} Hz")
        
        # Tabs for different plots
        tab1, tab2, tab3, tab4 = st.tabs(["Bearing Coefficients", "Bearing Locus", "Frequency Response", "Mode Shapes"])
        
        with tab1:
            self.plot_bearing_coefficients(analysis)
        
        with tab2:
            self.plot_bearing_locus(analysis)
        
        with tab3:
            self.plot_frequency_response(analysis)
        
        with tab4:
            self.plot_mode_shapes(analysis)
    
    def plot_bearing_coefficients(self, analysis):
        """Plot bearing stiffness and damping coefficients"""
        
        st.subheader("Bearing Stiffness and Damping Coefficients")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Bearing 1 stiffness
        axes[0, 0].plot(analysis.omega, analysis.k_m1_11, 'b-', label='K_yy', linewidth=2)
        axes[0, 0].plot(analysis.omega, analysis.k_m1_12, 'r-', label='K_yz', linewidth=2)
        axes[0, 0].plot(analysis.omega, analysis.k_m1_21, 'g-', label='K_zy', linewidth=2)
        axes[0, 0].plot(analysis.omega, analysis.k_m1_22, 'm-', label='K_zz', linewidth=2)
        axes[0, 0].set_ylim([-1e7, 1e7])
        axes[0, 0].set_xlim([0, analysis.omega[-1]])
        axes[0, 0].set_ylabel('Stiffness [N/m]')
        axes[0, 0].set_xlabel('Angular Velocity [rad/s]')
        axes[0, 0].legend()
        axes[0, 0].set_title(f'Bearing 1 Stiffness Coefficients ({analysis.bearing1_type.title()})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Bearing 2 stiffness
        axes[0, 1].plot(analysis.omega, analysis.k_m2_11, 'b-', label='K_yy', linewidth=2)
        axes[0, 1].plot(analysis.omega, analysis.k_m2_12, 'r-', label='K_yz', linewidth=2)
        axes[0, 1].plot(analysis.omega, analysis.k_m2_21, 'g-', label='K_zy', linewidth=2)
        axes[0, 1].plot(analysis.omega, analysis.k_m2_22, 'm-', label='K_zz', linewidth=2)
        axes[0, 1].set_ylim([-1e7, 1e7])
        axes[0, 1].set_xlim([0, analysis.omega[-1]])
        axes[0, 1].set_ylabel('Stiffness [N/m]')
        axes[0, 1].set_xlabel('Angular Velocity [rad/s]')
        axes[0, 1].legend()
        axes[0, 1].set_title(f'Bearing 2 Stiffness Coefficients ({analysis.bearing2_type.title()})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bearing 1 damping
        axes[1, 0].plot(analysis.omega, analysis.d_m1_11, 'b-', label='C_yy', linewidth=2)
        axes[1, 0].plot(analysis.omega, analysis.d_m1_12, 'r-', label='C_yz', linewidth=2)
        axes[1, 0].plot(analysis.omega, analysis.d_m1_21, 'g-', label='C_zy', linewidth=2)
        axes[1, 0].plot(analysis.omega, analysis.d_m1_22, 'm-', label='C_zz', linewidth=2)
        axes[1, 0].set_xlim([0, analysis.omega[-1]])
        axes[1, 0].set_ylabel('Damping [N·s/m]')
        axes[1, 0].set_xlabel('Angular Velocity [rad/s]')
        axes[1, 0].legend()
        axes[1, 0].set_title(f'Bearing 1 Damping Coefficients ({analysis.bearing1_type.title()})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Bearing 2 damping
        axes[1, 1].plot(analysis.omega, analysis.d_m2_11, 'b-', label='C_yy', linewidth=2)
        axes[1, 1].plot(analysis.omega, analysis.d_m2_12, 'r-', label='C_yz', linewidth=2)
        axes[1, 1].plot(analysis.omega, analysis.d_m2_21, 'g-', label='C_zy', linewidth=2)
        axes[1, 1].plot(analysis.omega, analysis.d_m2_22, 'm-', label='C_zz', linewidth=2)
        axes[1, 1].set_xlim([0, analysis.omega[-1]])
        axes[1, 1].set_ylabel('Damping [N·s/m]')
        axes[1, 1].set_xlabel('Angular Velocity [rad/s]')
        axes[1, 1].legend()
        axes[1, 1].set_title(f'Bearing 2 Damping Coefficients ({analysis.bearing2_type.title()})')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    def plot_bearing_locus(self, analysis):
        """Plot bearing locus or bearing information"""
        
        # Check if any bearings are journal type
        has_journal_bearings = (analysis.bearing1_type == "journal" or analysis.bearing2_type == "journal")
        
        if has_journal_bearings:
            st.subheader("Bearing Locus (Journal Center Trajectory)")
            
            # Create clearance circle
            theta = np.linspace(0, 2*np.pi, 1000)
            rho = 0.8
            Z_clearance = rho * np.cos(theta)
            Y_clearance = rho * np.sin(theta)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Plot journal bearing locus if applicable
            if analysis.bearing1_type == "journal":
                ExcZ1 = analysis.epsilon1 * np.sin(np.radians(analysis.phi1))
                ExcY1 = -analysis.epsilon1 * np.cos(np.radians(analysis.phi1))
                ax.plot(ExcZ1, ExcY1, 'b-', linewidth=3, label='Bearing 1 (Journal)', alpha=0.8)
            
            if analysis.bearing2_type == "journal":
                ExcZ2 = analysis.epsilon2 * np.sin(np.radians(analysis.phi2))
                ExcY2 = -analysis.epsilon2 * np.cos(np.radians(analysis.phi2))
                ax.plot(ExcZ2, ExcY2, 'r-', linewidth=3, label='Bearing 2 (Journal)', alpha=0.8)
            
            ax.plot(Z_clearance, Y_clearance, 'k--', linewidth=2, label='Clearance circle', alpha=0.6)
            
            ax.set_ylim((-0.8, 0.8))
            ax.set_ylabel('ε × cos(φ)', fontsize=12)
            ax.set_xlabel('ε × sin(φ)', fontsize=12)
            ax.set_xlim((-0.8, 0.8))
            ax.legend(fontsize=11)
            ax.set_title('Bearing Locus (Journal Bearings Only)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            st.pyplot(fig)
            plt.close()
        else:
            st.subheader("Bearing Information")
            st.info("📍 **Ball Bearings Used**: Ball bearings don't have a journal center trajectory like hydrodynamic bearings. They provide fixed stiffness and damping.")
            
            # Display ball bearing properties in a table
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Bearing 1 (Ball Bearing)**")
                bearing1_data = {
                    "Property": ["Stiffness Kxx", "Stiffness Kyy", "Damping Cxx", "Damping Cyy"],
                    "Value": [
                        f"{analysis.ball_bearing1_kxx/1e6:.1f} MN/m",
                        f"{analysis.ball_bearing1_kyy/1e6:.1f} MN/m", 
                        f"{analysis.ball_bearing1_cxx/1e3:.1f} kN·s/m",
                        f"{analysis.ball_bearing1_cyy/1e3:.1f} kN·s/m"
                    ]
                }
                st.table(bearing1_data)
            
            with col2:
                st.markdown("**Bearing 2 (Ball Bearing)**")
                bearing2_data = {
                    "Property": ["Stiffness Kxx", "Stiffness Kyy", "Damping Cxx", "Damping Cyy"],
                    "Value": [
                        f"{analysis.ball_bearing2_kxx/1e6:.1f} MN/m",
                        f"{analysis.ball_bearing2_kyy/1e6:.1f} MN/m",
                        f"{analysis.ball_bearing2_cxx/1e3:.1f} kN·s/m", 
                        f"{analysis.ball_bearing2_cyy/1e3:.1f} kN·s/m"
                    ]
                }
                st.table(bearing2_data)
    
    def plot_frequency_response(self, analysis):
        """Plot frequency response functions"""
        
        st.subheader("Frequency Response Functions")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convert angular velocity from rad/s to Hz
        frequency_hz = analysis.omega / (2 * np.pi)
        
        # FRF at bearing 1 (node 3)
        if analysis.X.shape[0] > 10:
            axes[0, 0].plot(frequency_hz, np.abs(analysis.X[8, :]) * 1000, 'b-', label='Y displacement', linewidth=2)
            axes[0, 0].plot(frequency_hz, np.abs(analysis.X[10, :]) * 1000, 'r-', label='Z displacement', linewidth=2)
        axes[0, 0].set_xlim([frequency_hz[0], frequency_hz[-1]])
        axes[0, 0].set_xlabel('Frequency [Hz]')
        axes[0, 0].set_ylabel('Amplitude [mm]')
        axes[0, 0].set_title('FRF - Bearing 1 Node')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # FRF at disk (node 6)
        if analysis.X.shape[0] > 22:
            axes[0, 1].plot(frequency_hz, np.abs(analysis.X[20, :]) * 1000, 'b-', label='Y displacement', linewidth=2)
            axes[0, 1].plot(frequency_hz, np.abs(analysis.X[22, :]) * 1000, 'r-', label='Z displacement', linewidth=2)
        axes[0, 1].set_xlim([frequency_hz[0], frequency_hz[-1]])
        axes[0, 1].set_xlabel('Frequency [Hz]')
        axes[0, 1].set_ylabel('Amplitude [mm]')
        axes[0, 1].set_title('FRF - Disk Node')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # FRF at bearing 2 (node 15)
        if analysis.X.shape[0] > 58:
            axes[1, 0].plot(frequency_hz, np.abs(analysis.X[56, :]) * 1000, 'b-', label='Y displacement', linewidth=2)
            axes[1, 0].plot(frequency_hz, np.abs(analysis.X[58, :]) * 1000, 'r-', label='Z displacement', linewidth=2)
        axes[1, 0].set_xlim([frequency_hz[0], frequency_hz[-1]])
        axes[1, 0].set_xlabel('Frequency [Hz]')
        axes[1, 0].set_ylabel('Amplitude [mm]')
        axes[1, 0].set_title('FRF - Bearing 2 Node')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Campbell diagram - convert critical frequencies to Hz
        axes[1, 1].plot(frequency_hz, frequency_hz, 'k-', linewidth=2, label='1X (Synchronous)')
        
        # Plot calculated natural frequency lines if available
        if hasattr(analysis, 'natural_freq_matrix'):
            colors = ['blue', 'orange', 'green', 'purple', 'brown']
            for mode_idx in range(min(5, analysis.natural_freq_matrix.shape[0])):
                nat_freq_line = analysis.natural_freq_matrix[mode_idx, :]
                # Only plot if we have valid data
                if np.any(nat_freq_line > 0):
                    nat_freq_hz = nat_freq_line / (2 * np.pi)
                    # Filter out zero values for cleaner plotting
                    valid_indices = nat_freq_line > 0
                    if np.any(valid_indices):
                        axes[1, 1].plot(frequency_hz[valid_indices], nat_freq_hz[valid_indices], 
                                       color=colors[mode_idx % len(colors)], linestyle='-', 
                                       linewidth=1.5, alpha=0.8, label=f'Mode {mode_idx+1}')
        
        # Plot calculated critical speeds as intersection markers
        if hasattr(analysis, 'critical_speeds') and len(analysis.critical_speeds) > 0:
            critical_labels = ['1st Critical', '2nd Critical', '3rd Critical']
            for i, critical_speed in enumerate(analysis.critical_speeds[:3]):
                critical_hz = critical_speed / (2 * np.pi)
                # Mark the critical speed as a point on the synchronous line
                axes[1, 1].plot(critical_hz, critical_hz, 'ro', markersize=8, 
                              label=critical_labels[i], markerfacecolor='red', 
                              markeredgecolor='darkred', markeredgewidth=2)
                # Add vertical line to show critical speed
                axes[1, 1].axvline(x=critical_hz, color='r', linestyle='--', alpha=0.5)
        else:
            # Fallback to hardcoded values
            axes[1, 1].axhline(y=160/(2*np.pi), color='r', linestyle='--', alpha=0.7, label='1st Critical')
            axes[1, 1].axhline(y=750/(2*np.pi), color='r', linestyle='--', alpha=0.7, label='2nd Critical')
            axes[1, 1].axhline(y=1460/(2*np.pi), color='r', linestyle='--', alpha=0.7, label='3rd Critical')
            
        axes[1, 1].set_xlabel('Rotor Speed [Hz]')
        axes[1, 1].set_ylabel('Natural Frequency [Hz]')
        axes[1, 1].set_title('Campbell Diagram')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim([frequency_hz[0], frequency_hz[-1]])
        axes[1, 1].set_ylim([frequency_hz[0], frequency_hz[-1]])
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    def plot_fe_model(self, analysis):
        """Plot 2D FE model with clean layout and separate force diagram"""
        
        st.subheader("Finite Element Model - System Layout")
        
        # Create two subplots: main layout and force diagram
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])
        
        # Get component positions (convert to mm for display)
        bearing1_pos_mm = getattr(self, 'bearing1_pos', 0.026) * 1000
        disk_pos_mm = getattr(self, 'disk_pos', 0.145) * 1000
        bearing2_pos_mm = getattr(self, 'bearing2_pos', 0.721) * 1000
        shaft_length_mm = analysis.le * 1000
        
        # Scale factors for visualization
        shaft_radius = analysis.de * 1000 / 2  # mm
        bearing_radius = analysis.D * 1000 / 2  # mm
        disk_radius = analysis.dd * 1000 / 2   # mm
        
        # ========== MAIN LAYOUT (TOP SUBPLOT) ==========
        # Draw shaft
        ax1.plot([0, shaft_length_mm], [shaft_radius, shaft_radius], 'k-', linewidth=4, label='Shaft')
        ax1.plot([0, shaft_length_mm], [-shaft_radius, -shaft_radius], 'k-', linewidth=4)
        ax1.fill_between([0, shaft_length_mm], [shaft_radius, shaft_radius], 
                        [-shaft_radius, -shaft_radius], alpha=0.3, color='gray')
        
        # Draw bearings with better visualization
        bearing_height = bearing_radius * 2
        bearing_width = 8
        
        # Bearing 1
        bearing1_rect = Rectangle((bearing1_pos_mm - bearing_width/2, -bearing_height/2), 
                                bearing_width, bearing_height, facecolor='steelblue', 
                                edgecolor='darkblue', linewidth=2, alpha=0.8, label='Bearings')
        ax1.add_patch(bearing1_rect)
        
        # Bearing support structure
        support_height = bearing_height * 0.6
        ax1.plot([bearing1_pos_mm - bearing_width, bearing1_pos_mm + bearing_width], 
                [-bearing_height/2 - support_height, -bearing_height/2 - support_height], 
                'k-', linewidth=3)
        ax1.plot([bearing1_pos_mm, bearing1_pos_mm], 
                [-bearing_height/2, -bearing_height/2 - support_height], 
                'k-', linewidth=2)
        
        ax1.text(bearing1_pos_mm, -bearing_height/2 - support_height - 20, 'Bearing 1', 
                ha='center', fontsize=12, fontweight='bold')
        
        # Bearing 2
        bearing2_rect = Rectangle((bearing2_pos_mm - bearing_width/2, -bearing_height/2), 
                                bearing_width, bearing_height, facecolor='steelblue', 
                                edgecolor='darkblue', linewidth=2, alpha=0.8)
        ax1.add_patch(bearing2_rect)
        
        # Bearing 2 support structure
        ax1.plot([bearing2_pos_mm - bearing_width, bearing2_pos_mm + bearing_width], 
                [-bearing_height/2 - support_height, -bearing_height/2 - support_height], 
                'k-', linewidth=3)
        ax1.plot([bearing2_pos_mm, bearing2_pos_mm], 
                [-bearing_height/2, -bearing_height/2 - support_height], 
                'k-', linewidth=2)
        
        ax1.text(bearing2_pos_mm, -bearing_height/2 - support_height - 20, 'Bearing 2', 
                ha='center', fontsize=12, fontweight='bold')
        
        # Draw disk with better visualization
        disk_width = analysis.ld * 1000  # disk thickness in mm
        disk_height = disk_radius * 2    # disk diameter in mm
        
        # Disk body
        disk_rect = Rectangle((disk_pos_mm - disk_width/2, -disk_height/2), 
                            disk_width, disk_height, facecolor='crimson', 
                            edgecolor='darkred', linewidth=2, alpha=0.8, label='Disk')
        ax1.add_patch(disk_rect)
        
        # Disk hub (smaller central part)
        hub_height = shaft_radius * 4
        hub_rect = Rectangle((disk_pos_mm - disk_width/2, -hub_height/2), 
                           disk_width, hub_height, facecolor='darkred', alpha=0.9)
        ax1.add_patch(hub_rect)
        
        ax1.text(disk_pos_mm, disk_height/2 + 15, 'Disk', ha='center', 
                fontsize=12, fontweight='bold')
        
        # Add clean dimension lines
        dimension_y = -bearing_height/2 - support_height - 50
        
        # Dimension line 1: Start to Bearing 1
        ax1.annotate('', xy=(bearing1_pos_mm, dimension_y), xytext=(0, dimension_y),
                   arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
        ax1.text(bearing1_pos_mm/2, dimension_y - 15, f'{bearing1_pos_mm:.0f} mm', 
                ha='center', fontsize=10, color='blue', fontweight='bold')
        
        # Dimension line 2: Bearing 1 to Disk
        ax1.annotate('', xy=(disk_pos_mm, dimension_y), xytext=(bearing1_pos_mm, dimension_y),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax1.text((bearing1_pos_mm + disk_pos_mm)/2, dimension_y - 15, 
                f'{disk_pos_mm - bearing1_pos_mm:.0f} mm', ha='center', fontsize=10, 
                color='red', fontweight='bold')
        
        # Dimension line 3: Disk to Bearing 2
        ax1.annotate('', xy=(bearing2_pos_mm, dimension_y), xytext=(disk_pos_mm, dimension_y),
                   arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax1.text((disk_pos_mm + bearing2_pos_mm)/2, dimension_y - 15, 
                f'{bearing2_pos_mm - disk_pos_mm:.0f} mm', ha='center', fontsize=10, 
                color='green', fontweight='bold')
        
        # Set main layout properties
        ax1.set_xlim(-30, shaft_length_mm + 30)
        y_max = disk_height/2 + 40
        y_min = dimension_y - 40
        ax1.set_ylim(y_min, y_max)
        ax1.set_xlabel('Axial Position [mm]', fontsize=12)
        ax1.set_ylabel('Radial Distance [mm]', fontsize=12)
        ax1.set_title('Rotordynamic System Layout', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # ========== FORCE DIAGRAM (BOTTOM SUBPLOT) ==========
        # Simplified force diagram
        force_scale = 50  # Smaller scale for cleaner arrows
        
        # Draw simplified beam for force diagram
        beam_y = 0
        ax2.plot([0, shaft_length_mm], [beam_y, beam_y], 'k-', linewidth=6, alpha=0.7)
        
        # Mark positions
        ax2.plot(bearing1_pos_mm, beam_y, 'bs', markersize=12, label='Bearing 1')
        ax2.plot(disk_pos_mm, beam_y, 'ro', markersize=12, label='Disk')
        ax2.plot(bearing2_pos_mm, beam_y, 'bs', markersize=12, label='Bearing 2')
        
        # Force arrows
        # Disk weight (downward)
        weight_arrow_length = analysis.W / force_scale
        ax2.arrow(disk_pos_mm, beam_y, 0, -weight_arrow_length, head_width=15, head_length=8,
                fc='red', ec='red', linewidth=3, alpha=0.8)
        ax2.text(disk_pos_mm, -weight_arrow_length - 20, f'W = {analysis.W:.1f} N', 
                ha='center', fontsize=11, color='red', fontweight='bold')
        
        # Bearing reactions (upward)
        fm1_arrow_length = analysis.FM1 / force_scale
        ax2.arrow(bearing1_pos_mm, beam_y, 0, fm1_arrow_length, head_width=15, head_length=8,
                fc='blue', ec='blue', linewidth=3, alpha=0.8)
        ax2.text(bearing1_pos_mm, fm1_arrow_length + 10, f'R₁ = {analysis.FM1:.1f} N', 
                ha='center', fontsize=11, color='blue', fontweight='bold')
        
        fm2_arrow_length = analysis.FM2 / force_scale
        ax2.arrow(bearing2_pos_mm, beam_y, 0, fm2_arrow_length, head_width=15, head_length=8,
                fc='blue', ec='blue', linewidth=3, alpha=0.8)
        ax2.text(bearing2_pos_mm, fm2_arrow_length + 10, f'R₂ = {analysis.FM2:.1f} N', 
                ha='center', fontsize=11, color='blue', fontweight='bold')
        
        # Force diagram properties
        ax2.set_xlim(-30, shaft_length_mm + 30)
        max_force_arrow = max(weight_arrow_length, fm1_arrow_length, fm2_arrow_length)
        ax2.set_ylim(-max_force_arrow - 40, max_force_arrow + 40)
        ax2.set_xlabel('Position [mm]', fontsize=12)
        ax2.set_ylabel('Force [N]', fontsize=12)
        ax2.set_title('Static Force Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linewidth=1, alpha=0.5)
        
        # Add force balance equation
        force_sum = analysis.FM1 + analysis.FM2 - analysis.W
        ax2.text(shaft_length_mm/2, -max_force_arrow - 25, 
                f'Equilibrium Check: ΣF = R₁ + R₂ - W = {force_sum:.3f} N ≈ 0', 
                ha='center', fontsize=10, style='italic', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    def plot_mode_shapes(self, analysis):
        """Plot mode shapes at different frequencies"""
        
        st.subheader("Mode Shapes at Critical Speeds")
        
        # Use calculated critical speeds or fallback to default
        if hasattr(analysis, 'critical_speeds') and len(analysis.critical_speeds) > 0:
            speeds = analysis.critical_speeds[:3]  # Use calculated critical speeds
        else:
            speeds = [160, 750, 1460]  # Fallback to default values
        
        # Find indices for specific frequencies
        indices = []
        for speed in speeds:
            idx = np.argmin(np.abs(analysis.omega - speed))
            indices.append(idx)
        
        # Create subplots based on number of critical speeds
        num_plots = min(3, len(speeds))
        fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
        if num_plots == 1:
            axes = [axes]  # Make it iterable for single plot
        
        for i, (speed, idx) in enumerate(zip(speeds[:num_plots], indices[:num_plots])):
            if idx < analysis.X.shape[1] and analysis.X.shape[0] > 4:
                y_displacements = analysis.X[::4, idx].real  # Every 4th element (Y displacements)
                axes[i].plot(analysis.coord[:, 1], y_displacements, 'b-o', linewidth=3, markersize=6, label='Y deformation')
                axes[i].plot(analysis.coord[:, 1], analysis.coord[:, 2], 'k--', marker='s', markersize=4, label='Initial state', alpha=0.6)
            
            # Convert speed to Hz
            speed_hz = speed / (2 * np.pi)
            
            axes[i].legend()
            axes[i].set_xlabel('Axial Position [m]')
            axes[i].set_ylabel('Amplitude [m]')
            axes[i].set_title(f'Mode Shape at Ω = {speed} rad/s ({speed_hz:.1f} Hz)')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    def run(self):
        """Main application runner"""
        
        # Header
        st.title("Rotordynamic Analysis Tool")
        st.markdown("**Advanced finite element analysis of rotating shaft-disk-bearing systems**")
        st.markdown("---")
        
        # Get parameters from sidebar
        params = self.create_sidebar()
        
        # Update analysis object
        self.update_analysis_parameters(params)
        
        # Main content area - Clean interface with just the run button
        self.run_analysis_button()
        
        st.markdown("---")
        
        # Display FE Model
        if st.session_state.get('analysis_complete', False):
            analysis = st.session_state['analysis_object']
            self.plot_fe_model(analysis)
            st.markdown("---")
        
        # Display results
        self.display_results()

# Initialize and run the app
if __name__ == "__main__":
    app = StreamlitRotordynamicApp()
    app.run() 
