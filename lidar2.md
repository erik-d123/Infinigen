LiDAR Model and Simulation Refinement for
Indoor SLAM
1. LiDAR Sensor Model Selection (Vertical FoV & Usage Trends)
Recommendation: Switch to simulating an Ouster OS0-128 (or similar wide-FOV LiDAR) for indoor
scenes. The current Ouster OS1-128 model provides a ±22.5° vertical field of view (45° total), which may
miss floor and ceiling features in indoor environments. Recent robotics and SLAM literature favor sensors
with broader vertical coverage to capture low and high obstacles in tight spaces. The Ouster OS0-128 offers
a 90° vertical FoV (±45°) – double that of the OS1 – specifically to “see floor to ceiling” in warehouses and
indoor settings 1 2
. This wide FoV ensures the LiDAR captures entire rooms, from ground clutter to
3
overhead structures, in a single spin .
In contrast, many legacy sensors have much narrower vertical spans. For example, the Velodyne VLP-16
(“Puck”) used in the TartanAir dataset has only ~30° vertical FoV 4
, and even the Velodyne HDL-32E covers
~41° (from about –30° to +10°) – meaning they can miss ceilings or require tilting the sensor. Newer 32-
beam units like the Hesai XT32 likewise cover only ~31° vertically 5
. By adopting the OS0-128, you align
with the trend toward high-resolution, wide-view LiDARs in indoor robotics. Ouster themselves note
the OS0 is “best suited” for indoor/mobile robots, trading range for a full 90° FoV and 128 channels of
resolution 6 2 6
. The shorter effective range (~50 m at 80% reflectivity ) is not a concern for indoor
use, and the dense point cloud (2.6 million points/sec) provides detailed geometry even at close range
7
8
. In summary, switching to an OS0-128 model is recommended to improve vertical coverage and
match the sensor setups common in recent SLAM research (e.g. autonomous warehouse robots using OS0
2
). This choice will ensure your synthetic data is both comprehensive and comparable to modern real-
world datasets.
(If OS0-128 hardware specs are adopted, update the simulation’s vertical beam angles to ±45°. The horizontal
resolution and spin rate can remain similar; OS0 and OS1 share up to 2048 azimuthal points at 10 Hz 9
may also consider offering a Velodyne VLP-16 mode for compatibility with older benchmarks 4
. You
, but the OS0-128
will provide more robust ground-truth coverage for new experiments.)
2. Validation of Simulation Equations & Parameters (Models and
References)
Your LiDAR simulation already employs physically-grounded models. Below we review each component –
intensity, reflectivity, range noise, beam divergence, and timing – providing canonical equations or data
sources and noting any adjustments to consider:
•
Radiometric Intensity & Reflectivity: The simulated intensity $I$ is computed from material
reflectance and distance using an inverse-square law, modulated by surface properties and
incidence angle. This is consistent with the standard LiDAR range equation (analogous to the radar
1
equation). For a Lambertian target at normal incidence, received power $P_r$ is proportional to the
target reflectivity $\rho$ divided by range squared 10
. In your code: you form a “reflectivity” term
from the Blender Principled BSDF (diffuse + specular reflectance) and apply intensity =
11
reflectivity / (distance^p) (with $p=2$ by default) . This matches the free-space path
loss ($1/R^2$) included in other simulators – e.g. Blensor accounts for material reflectance and FSPL in
its intensity model 12
. Crucially, you also include the cosine of incidence in the reflectivity: diffuse
Lambertian returns scale by $\cos\theta$ (surface normal vs. beam) 13
. This is physically correct –
real LiDAR intensities drop at grazing angles, which is why intensity calibration methods explicitly
model incidence-angle dependence 14 15 16
. Your use of Schlick’s Fresnel for specular highlights
17
and combining it with diffuse reflectance is an advanced, physically-based approach.
Recommended references: The CarMaker LiDAR model and others compute point intensity from
material properties, incidence angle, and propagation loss just as you do 18 19
. In effect, your
intensity equation is well-founded. We suggest documenting it as:
$$ I = \frac{\rho_\mathrm{eff}(\text{material}, \theta)}{R^2}, $$
where $\rho_\mathrm{eff}$ is the effective reflectivity after accounting for diffuse albedo (scaled by $
\cos\theta$) and any specular/Fresnel component. This highlights that intensity is essentially reflectance (or
“calibrated reflectivity” 20
) attenuated by $1/R^2$. Citations: The inverse-square law and Lambertian
cosine fall-off are textbook LiDAR physics 10
, and they are implemented in simulation frameworks (e.g.
DYNA4, AURELION) which factor in surface reflectivity and angle 21
. No major corrections are needed here
– your parameters (e.g. enforcing a minimum 2% albedo to avoid zero intensity 13
) are reasonable. Just
ensure that if “auto_exposure” is used, it’s clearly explained, or consider outputting a calibrated reflectivity
22
value (which you do as the reflectivity channel) for research use .
•
Range Noise Model: You add Gaussian noise to the distance measurement with a standard
deviation that grows linearly with range (e.g. $\sigma_r = a + b\cdot R$, with defaults $a=0.01$ m,
$b=0.001$ m/m) 23 24
. This is a sensible empirical model. Real LiDAR range accuracy is on the
order of centimeters, typically around 2–3 cm (1$\sigma$) in good conditions. For instance, one
study notes a real sensor’s spec is ±2 cm 25 5
, and Hesai advertises ~±1 cm accuracy for the XT32 .
Ouster’s datasheet even boasts “millimeter-level precision” at close range 7
. Your chosen
parameters (1 cm base noise, +1 mm per 1 m) yield ~2 cm noise at 10 m and ~11 cm at 100 m, which
is slightly higher at long range but still plausible. Recommendation: Cite the sensor datasheets or
studies for these values – e.g., Velodyne Puck: ±3 cm, Ouster OS0: ~5 mm at <5 m – to justify the noise
profile. If focusing on indoor ranges (say <30 m), your linear model will keep noise ~<4 cm, which is
reasonable. You might consider reducing the base $\sigma$ to ~5 mm if simulating an OS0 (to
reflect its finer precision at short range 26
), but this is a minor tweak. Overall, the Gaussian range
noise assumption is standard, as also used by other simulators (often simply a fixed cm-level error
budget) 25
. Just be sure to document that the random seed and distribution are known, in case
researchers want to reproduce or disable the noise.
•
Beam Divergence & Footprint: Real LiDAR laser beams have a non-zero divergence (e.g. on the
order of 0.3–0.9 mrad for many sensors). This means the laser footprint expands with distance,
which can affect returns (especially at edges or small objects). Your simulation addresses this subtly:
you apply an incidence angle exponent ($\cos^k$) to specular returns 17
, essentially damping
specular intensity for grazing hits – described in code as a “footprint effect” correction. This is a
reasonable heuristic (glancing beams illuminate a larger spot, lowering peak reflected power). In
2
•
•
high-fidelity models, beam divergence is sometimes explicitly simulated by casting multiple sub-rays
or modeling a Gaussian beam profile 27
. For example, Goodin et al. (VANE simulator) included beam
divergence with a Gaussian intensity profile across the beam cross-section 27
. The open-source
HELIOS simulator also accounts for beam spot size growth and scanner optics 28
. Your approach
is simpler but should be sufficient for most indoor targets, given typical divergence (~0.005–0.01°
per beam) causes only minor intensity falloff within common ranges. Recommendation: If a surface
is extremely smooth and mirror-like, your specular model already handles the main effect (Fresnel
reflectance and angle). For completeness, you could note the assumed beam divergence of the
chosen model (e.g. “assuming ~0.5 mrad beam divergence resulting in ~5 cm spot at 100 m”). If future
work demands it, one could sample a few rays per beam to simulate partial hits (as in vegetation or
edges), but this adds cost. Given indoor scenes of mostly solid surfaces, your current divergence
treatment is acceptable. Just cite that this aligns with prior simulations that include divergence
effects 29
and mention that the chosen LiDAR’s datasheet beam divergence is used as a parameter
(if available).
Scanning Pattern and Timing: Your implementation generates points according to a specific
vertical angular distribution and simulates the rotational scan over time. This is important:
spinning LiDARs operate with a rolling shutter, so points in one revolution have different
timestamps. You have already included this realism – e.g. you define a spin rate (RPM) and have
continuous_spin=True with rolling_shutter=True to timestamp points as the sensor
rotates 30 30
. The OS1/OS0 typically run at 10 Hz (600 RPM), which you use as default . This
matches real sensors (Velodyne, Ouster) and introduces motion distortion effects if the sensor or
scene moves during a scan. Recommendation: Clearly present that one frame of LiDAR data
corresponds to (for example) 100 ms of sensor rotation. Each point’s timestamp can be derived
along that interval. This detail is supported by how actual sensor data is produced – for instance, the
KITTI benchmark noted the Velodyne spins continuously, requiring motion compensation for
accuracy 31
. In your presentation, you might illustrate the azimuth/angle mapping: e.g., “128
beams from +45° to –45° (if OS0) and ~2048 columns over 360°, sampled in 0.1 s per revolution.”
Provide references that such scanning patterns are standard; for example, Ouster’s user forum
confirms all their sensors use 128 channels and up to 2048 horizontal steps per spin 32
. No changes
are needed to your timing logic – you correctly emulate a rotating lidar. Just ensure the vertical
angle list is updated if you change models (your current OS1-128 uses a linear –22.5° to +22.5°
spread 33
; an OS0-128 would be –45° to +45°). Manufacturers often supply exact beam elevation
angles (e.g. for Ouster: Uniform vs. non-uniform distributions), so using those ensures fidelity.
Other Considerations (Multiple Returns, etc.): It’s worth noting your simulator can handle multi-
echo returns (e.g. enable_secondary in the config). Real LiDARs often report two returns per
beam (e.g. Ouster’s dual return mode 34
). Your implementation launches a secondary ray when a
beam only partially hits a surface (based on a transmissivity threshold) 35
. This is a sophisticated
feature – e.g. a laser hitting a pane of glass could yield a weak first return on the glass and a second
return on an object behind it. If you plan to present this, cite sensor datasheets or papers that
mention multi-return capability (Ouster and Velodyne document this). For instance, the Ouster L2X
chip processes strongest and second-strongest returns for each channel 34
. Your simulation’s
approach (continue the ray with residual energy/transmittance) is analogous to how Blensor and
others simulate multiple returns in semi-transparent media 19
. No fundamental issues here; just
verify your default (you have it off by default) and mention it as a toggle for advanced use.
3
In summary, each aspect of your LiDAR ground-truth generator is grounded in known models. The
intensity model follows the LiDAR equation (with range and angle factors) 10 14
, the noise model
reflects real sensor precision (a few cm) 25 5
, the beam geometry and scan timing mirror actual
hardware behavior 27 30
. The key references to cite in your write-up will include sensor manuals (for FOV,
range, accuracy) and prior simulation works: e.g. Blensor’s paper for radiometric modeling 12
, CARLA or
OSI documentation for intensity and noise modeling 21
, and any academic papers on LiDAR realism
(Goodin et al. for divergence 27
, etc.). By attributing each equation to such sources, you’ll lend credibility to
your implementation. Based on our review, no major corrections are needed – just ensure the sensor
model choice (OS0 vs OS1) is decided now, as it affects the vertical angle array and range settings used.
With an OS0-128 and the validated equations above, your LiDAR ground-truth generator will be well-
supported by both literature and industry standards, ready for a rigorous research presentation.
Sources: The justification for switching to an OS0-128 is supported by Ouster’s specifications 1 2
and
industry comparisons 36
. Equations for LiDAR intensity and noise follow established physics and simulator
practices 10 12 18 25 7
, and the chosen parameters fall within real sensor performance ranges . The
inclusion of incidence angle, beam divergence, and rolling shutter timing in the simulation aligns with
recommendations from prior work on high-fidelity LiDAR models 27 14
. All references are listed below for
each aspect discussed.
1 2 6 9 32 34
Choosing a LiDAR sensor for your project - Ouster lidar forum by General Laser
https://forum.ouster.at/d/137-choosing-a-lidar-sensor-for-your-project
3 7 8 26
The OS0 wide-view lidar sensor: deep dive | Ouster
https://ouster.com/insights/blog/the-os0-wide-view-lidar-sensor-deep-dive
4
Modalities — TartanAir documentation
https://tartanair.org/modalities.html
5
HESAI XT32M1X LiDAR‑Sensor 360° x 31° | 32 Channels - Epotronic
https://epotronic.com/eng/HESAI-XT32M1X-High-Precision-3600-x-310-Mid-Range-80-m-LiDAR-Sensor-32-Channels/HXT002
10
Title
https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/static/hc/resources/W0004/lidar_webinar_12.6.17.pdf
11 13 16 17 22
lidar_engine.py
https://github.com/erik-d123/Infinigen/blob/45782c659ee3511779ff3756e1d564edad6e0bfb/infinigen/lidar/lidar_engine.py
12 15 18 19 21 25 27 28 29
Development of High-Fidelity Automotive LiDAR Sensor Model with
Standardized Interfaces
https://www.mdpi.com/1424-8220/22/19/7556
14 20
Reflectivity Is All You Need!: Advancing LiDAR Semantic Segmentation
https://arxiv.org/html/2403.13188v1
23 24 30 31 33 35
lidar_generator_backup.py
https://github.com/erik-d123/Infinigen/blob/45782c659ee3511779ff3756e1d564edad6e0bfb/backup_files/
lidar_generator_backup.py
36
Not All LiDAR Sensors Are Equal: Key Differences Explained
https://insights.outsight.ai/not-all-lidar-are-created-equal/
4