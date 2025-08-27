import numpy as np
import matplotlib.pyplot as plt

class PipePlot:
    def __init__(self, outer_radius, inner_radius, length, highlight_distance, highlight_width):
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.length = length
        self.highlight_distance = highlight_distance
        self.highlight_width = highlight_width
        self.theta = np.linspace(0, 2*np.pi, 200)
        self.z = np.linspace(0, length, 200)
        self.Theta, self.Z = np.meshgrid(self.theta, self.z)
        self.Xo = outer_radius * np.cos(self.Theta)
        self.Yo = outer_radius * np.sin(self.Theta)
        self.Zo = self.Z
        self.Xi = inner_radius * np.cos(self.Theta)
        self.Yi = inner_radius * np.sin(self.Theta)
        self.Zi = self.Z
        R = np.linspace(inner_radius, outer_radius, 200)
        self.Theta_cap, self.R_cap = np.meshgrid(self.theta, R)
        self.Xcap = self.R_cap * np.cos(self.Theta_cap)
        self.Ycap = self.R_cap * np.sin(self.Theta_cap)
        self.Zcap0 = np.zeros_like(self.Xcap)
        self.ZcapL = np.full_like(self.Xcap, length)
        self.blue = np.array([0, 0, 1, 1])
        self.red = np.array([1, 0, 0, 1])

    def hotspot(self, z):
        return np.exp(-((z - self.highlight_distance)**2) / (2 * self.highlight_width**2))

    def mix_color(self, z_array):
        w = self.hotspot(z_array)
        return (1 - w)[..., None]*self.blue + w[..., None]*self.red

    def plot(self):
        colors_outer = self.mix_color(self.Zo)
        colors_inner = self.mix_color(self.Zi)
        colors_cap0 = self.mix_color(self.Zcap0)
        colors_capL = self.mix_color(self.ZcapL)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(self.Xo, self.Yo, self.Zo, facecolors=colors_outer, linewidth=0, antialiased=False, shade=False)
        ax.plot_surface(self.Xi, self.Yi, self.Zi, facecolors=colors_inner, linewidth=0, antialiased=False, shade=False)
        ax.plot_surface(self.Xcap, self.Ycap, self.Zcap0, facecolors=colors_cap0, linewidth=0, antialiased=False, shade=False)
        ax.plot_surface(self.Xcap, self.Ycap, self.ZcapL, facecolors=colors_capL, linewidth=0, antialiased=False, shade=False)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Pipe with Local Red Gradient at Highlight Distance')
        max_range = max(self.Xo.max()-self.Xo.min(), self.Yo.max()-self.Yo.min(), self.length)
        ax.set_xlim(-max_range/2, max_range/2)
        ax.set_ylim(-max_range/2, max_range/2)
        ax.set_zlim(0, self.length)
        plt.tight_layout()
        plt.show()
if __name__ == '__main__':
        
    pipe = PipePlot(0.1524, 0.14, 1.0, 0.25, 0.1)
    pipe.plot()
