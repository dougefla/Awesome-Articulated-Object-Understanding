# Awesome Articulated Objects Understanding
A curated list of resources for articulated objects understanding.

## Articulated Objects Reconstruction
### PARIS: Part-level Reconstruction and Motion Analysis for Articulated Objects [ICCV 2023]
[📄 Paper](https://arxiv.org/abs/2308.07391) | [🌐 Project Page](https://3dlg-hcvc.github.io/paris/) | [💻 Code](https://github.com/3dlg-hcvc/paris)
- Keywords: Self-supervised, NeRF
- Input: Two sets of multi-view images
- Dataset: PartNet-Mobility, MultiScan
<details span>
<summary><b>Abstract</b></summary>
<br>

We design a self-supervised approach without relying any 3D supervision, semantic or motion annotations. The key idea is that we separate the parts from two-state observations by leveraging motion as a cue. Since the motion accounts for the inconsistency between two states, we optimize the motion parameters by registering the moving parts from the input states t to a canonical state t*. During registration, the component that agrees with the transformation is extracted as the movable part. And the one remaining still is extracted as the static part.
</details>

### Ditto in the house: Building articulation models of indoor scenes through interactive perception [ICRA 2023]
[📄 Paper](https://arxiv.org/abs/2302.01295) | [🌐 Project Page](https://ut-austin-rpl.github.io/HouseDitto/) | [💻 Code](https://github.com/UT-Austin-RPL/HouseDitto)
- Keywords: Articulation, Supervised, PointNet++
- Input: Two point clouds
- Dataset: CubiCasa5K
<details span>
<summary><b>Abstract</b></summary>
<br>

Our approach, named Ditto in the House, discovers possible articulated objects through affordance prediction, interacts with these objects to produce articulated motions, and infers the articulation properties from the visual observations before and after each interaction. The approach consists of two stages — affordance prediction and articulation inference. During affordance prediction, we pass the static scene point cloud into the affordance network and predict the scene-level affordance map. Then, the robot interacts with the object based on those contact points. During articulation inference, we feed the point cloud observations before and after each interaction into the articulation model network to obtain articulation estimation. By aggregating the estimated articulation models, we build the articulation models of the entire scene.
</details>

### Semi-Weakly Supervised Object Kinematic Motion Prediction [CVPR 2023]
[📄 Paper](https://arxiv.org/abs/2303.17774) | [🌐 Project Page](https://vcc.tech/research/2023/SWMP) | [💻 Code](https://github.com/GengxinLiu/SWMP)
- Keywords: Articulation, Semi-supervised, GNN
- Input: Single Point cloud + segmentation
- Dataset: PartNet-Mobility, PartNet
<details span>
<summary><b>Abstract</b></summary>
<br>

In this paper, we tackle the task of object kinematic motion prediction problem in a semi-weakly supervised manner. Our key observations are two-fold. First, although 3D dataset with fully annotated motion labels is limited, there are existing datasets and methods for object part semantic segmentation at large scale. Second, semantic part segmentation and mobile part segmentation is not always consistent but it is possible to detect the mobile parts from the underlying 3D structure. Towards this end, we propose a graph neural network to learn the map between hierarchical part-level segmentation and mobile parts parameters, which are further refined based on geometric alignment. This network can be first trained on PartNet-Mobility dataset with fully labeled mobility information and then applied on PartNet dataset with fine-grained and hierarchical part-level segmentation. The network predictions yield a large scale of 3D objects with pseudo labeled mobility information and can further be used for weakly-supervised learning with pre-existing segmentation. Our experiments show there are significant performance boosts with the augmented data for previous method designed for kinematic motion prediction on 3D partial scans.
</details>

### CARTO: Category and Joint Agnostic Reconstruction of ARTiculated Objects [CVPR 2023]
[📄 Paper](https://arxiv.org/abs/2303.15782) | [🌐 Project Page](http://carto.cs.uni-freiburg.de/) | [💻 Code](https://github.com/robot-learning-freiburg/CARTO)
- Keywords: Articulation, Reconstruction, Supervised
- Input: Single RGB image
- Dataset: PartNet-Mobility
<details span>
<summary><b>Abstract</b></summary>
<br>

We present CARTO, a novel approach for reconstructing multiple articulated objects from a single stereo RGB observation. We use implicit object-centric representations and learn a single geometry and articulation decoder for multiple object categories. Despite training on multiple categories, our decoder achieves a comparable reconstruction accuracy to methods that train bespoke decoders separately for each category. Combined with our stereo image encoder we infer the 3D shape, 6D pose, size, joint type, and the joint state of multiple unknown objects in a single forward pass. Our method achieves a 20.4% absolute improvement in mAP 3D IOU50 for novel instances when compared to a two-stage pipeline. Inference time is fast and can run on a NVIDIA TITAN XP GPU at 1 HZ for eight or less objects present. While only trained on simulated data, CARTO transfers to real-world object instances. Code and evaluation data is linked below.
</details>

### CA2T-Net: Category-Agnostic 3D Articulation Transfer from Single Image [Arxiv 2023]
[📄 Paper](https://arxiv.org/abs/2301.02232)
- Keywords: Articulation, Transfer, Supervised, ResNet
- Input: RGB, Mesh
- Dataset: PartNet-Mobility
<details span>
<summary><b>Abstract</b></summary>
<br>

We present a neural network approach to transfer the motion from a single image of an articulated object to a rest-state (i.e., unarticulated) 3D model. Our network learns to predict the object's pose, part segmentation, and corresponding motion parameters to reproduce the articulation shown in the input image. The network is composed of three distinct branches that take a shared joint image-shape embedding and is trained end-to-end. Unlike previous methods, our approach is independent of the topology of the object and can work with objects from arbitrary categories. Our method, trained with only synthetic data, can be used to automatically animate a mesh, infer motion from real images, and transfer articulation to functionally similar but geometrically distinct 3D models at test time.
</details>

### Command-driven Articulated Object Understanding and Manipulation [CVPR 2023]
[📄 Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Chu_Command-Driven_Articulated_Object_Understanding_and_Manipulation_CVPR_2023_paper.html)
- Keywords: Articulation, Supervised, U-Net
- Input: Single Point cloud
- Dataset: Shape2Motion, PartNet-Mobility
<details span>
<summary><b>Abstract</b></summary>
<br>

We present Cart, a new approach towards articulated-object manipulations by human commands. Beyond the existing work that focuses on inferring articulation structures, we further support manipulating articulated shapes to align them subject to simple command templates. The key of Cart is to utilize the prediction of object structures to connect visual observations with user commands for effective manipulations. It is achieved by encoding command messages for motion prediction and a test-time adaptation to adjust the amount of movement from only command supervision. For a rich variety of object categories, Cart can accurately manipulate object shapes and outperform the state-of-the-art approaches in understanding the inherent articulation structures. Also, it can well generalize to unseen object categories and real-world objects. We hope Cart could open new directions for instructing machines to operate articulated objects.
</details>

### MultiScan: Scalable RGBD scanning for 3D environments with articulated objects [NIPS 2022]
[📄 Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3b3a83a5d86e1d424daefed43d998079-Abstract-Conference.html) | [💻 Code](https://github.com/smartscenes/multiscan) | [🌐 Project Page](https://3dlg-hcvc.github.io/multiscan/#/)
- Keywords: Dataset, MultiScan
- Input: Not specified
<details span>
<summary><b>Abstract</b></summary>
<br>

We introduce MultiScan, a scalable RGBD dataset construction pipeline leveraging commodity mobile devices to scan indoor scenes with articulated objects and web-based semantic annotation interfaces to efficiently annotate object and part semantics and part mobility parameters. We use this pipeline to collect 273 scans of 117 indoor scenes containing 10957 objects and 5129 parts. The resulting MultiScan dataset provides RGBD streams with per-frame camera poses, textured 3D surface meshes, richly annotated part-level and object-level semantic labels, and part mobility parameters. We validate our dataset on instance segmentation and part mobility estimation tasks and benchmark methods for these tasks from prior work. Our experiments show that part segmentation and mobility estimation in real 3D scenes remain challenging despite recent progress in 3D object segmentation.
</details>

### AKB-48: A real-world articulated object knowledge base [CVPR 2022]
[📄 Paper](https://arxiv.org/abs/2202.08432) | [🌐 Project Page](https://liuliu66.github.io/articulationobjects/)
- Keywords: Dataset, Articulation, Reconstruction, Supervised
- Input: Single RGB-D image
- Dataset: AKB-48
<details span>
<summary><b>Abstract</b></summary>
<br>

Human life is populated with articulated objects. A comprehensive understanding of articulated objects, namely appearance, structure, physics property, and semantics, will benefit many research communities. As current articulated object understanding solutions are usually based on synthetic object dataset with CAD models without physics properties, which prevent satisfied generalization from simulation to real-world applications in visual and robotics tasks. To bridge the gap, we present AKB-48: a large-scale Articulated object Knowledge Base which consists of 2,037 real-world 3D articulated object models of 48 categories. Each object is described by a knowledge graph ArtiKG. To build the AKB-48, we present a fast articulation knowledge modeling (FArM) pipeline, which can fulfill the ArtiKG for an articulated object within 10-15 minutes, and largely reduce the cost for object modeling in the real world. Using our dataset, we propose AKBNet, a novel integral pipeline for Category-level Visual Articulation Manipulation (C-VAM) task, in which we benchmark three sub-tasks, namely pose estimation, object reconstruction and manipulation.
</details>

### FlowBot3D: Learning 3D Articulation Flow to Manipulate Articulated Objects [RSS 2022]
[📄 Paper](https://arxiv.org/abs/2205.04382) | [🌐 Project Page](https://sites.google.com/view/articulated-flowbot-3d/home) | [💻 Code](https://github.com/r-pad/flowbot3d)
- Keywords: Articulation, Supervised, PointNet++
- Input: Single Point cloud
- Dataset: PartNet-Mobility
<details span>
<summary><b>Abstract</b></summary>
<br>

We explore a novel method to perceive and manipulate 3D articulated objects that generalizes to enable a robot to articulate unseen classes of objects. We propose a vision-based system that learns to predict the potential motions of the parts of a variety of articulated objects to guide downstream motion planning of the system to articulate the objects. To predict the object motions, we train a neural network to output a dense vector field representing the point-wise motion direction of the points in the point cloud under articulation. We then deploy an analytical motion planner based on this vector field to achieve a policy that yields maximum articulation. We train the vision system entirely in simulation, and we demonstrate the capability of our system to generalize to unseen object instances and novel categories in both simulation and the real world, deploying our policy on a Sawyer robot with no finetuning. Results show that our system achieves state-of-the-art performance in both simulated and real-world experiments.
</details>

### Category-Independent Articulated Object Tracking with Factor Graphs [IROS 2022]
[📄 Paper](https://arxiv.org/abs/2205.03721) | [🌐 Project Page](https://sites.google.com/view/category-independentarticulate/category-independent-articulated-object-tracking-with-factor-graphs) | [💻 Code](https://github.com/SuperN1ck/cat-ind-fg)
- Keywords: Articulation, Supervised+Factor Graph
- Input: RGB-D sequence
- Dataset: PartNet-Mobility
<details span>
<summary><b>Abstract</b></summary>
<br>

Robots deployed in human-centric environments may need to manipulate a diverse range of articulated objects, such as doors, dishwashers, and cabinets. Articulated objects often come with unexpected articulation mechanisms that are inconsistent with categorical priors: for example, a drawer might rotate about a hinge joint instead of sliding open. We propose a category-independent framework for predicting the articulation models of unknown objects from sequences of RGB-D images. The prediction is performed by a two-step process: first, a visual perception module tracks object part poses from raw images, and second, a factor graph takes these poses and infers the articulation model including the current configuration between the parts as a 6D twist. We also propose a manipulation-oriented metric to evaluate predicted joint twists in terms of how well a compliant robot controller would be able to manipulate the articulated object given the predicted twist. We demonstrate that our visual perception and factor graph modules outperform baselines on simulated data and show the applicability of our factor graph on real world data.
</details>

### Ditto: Building Digital Twins of Articulated Objects from Interaction [CVPR 2022]
[📄 Paper](https://arxiv.org/abs/2202.08227) | [🌐 Project Page](https://rpl.cs.utexas.edu/publications/2022/06/19/jiang-cvpr22-ditto/) | [💻 Code](https://github.com/UT-Austin-RPL/Ditto)
- Keywords: Articulation, Supervised, PointNet++, NeRF
- Input: Two point clouds
- Dataset: Shape2Motion
<details span>
<summary><b>Abstract</b></summary>
<br>

Digitizing physical objects into the virtual world has the potential to unlock new research and applications in embodied AI and mixed reality. This work focuses on recreating interactive digital twins of real-world articulated objects, which can be directly imported into virtual environments. We introduce Ditto to learn articulation model estimation and 3D geometry reconstruction of an articulated object through interactive perception. Given a pair of visual observations of an articulated object before and after interaction, Ditto reconstructs part-level geometry and estimates the articulation model of the object. We employ implicit neural representations for joint geometry and articulation modeling. Our experiments show that Ditto effectively builds digital twins of articulated objects in a category-agnostic way. We also apply Ditto to real-world objects and deploy the recreated digital twins in physical simulation.
</details>

### OPD: Single-view 3D Openable Part Detection [ECCV 2022]
[📄 Paper](https://arxiv.org/abs/2203.16421) | [🌐 Project Page](https://3dlg-hcvc.github.io/OPD/) | [💻 Code](https://github.com/3dlg-hcvc/OPD)
- Keywords: Articulation, Supervised, MaskRCNN
- Input: Single Point cloud
- Dataset: OPDSynth, OPDReal
<details span>
<summary><b>Abstract</b></summary>
<br>

We address the task of predicting what parts of an object can open and how they move when they do so. The input is a single image of an object, and as output, we detect what parts of the object can open, and the motion parameters describing the articulation of each openable part. To tackle this task, we create two datasets of 3D objects: OPDSynth based on existing synthetic objects, and OPDReal based on RGBD reconstructions of real objects. We then design OPDRCNN, a neural architecture that detects openable parts and predicts their motion parameters. Our experiments show that this is a challenging task especially when considering generalization across object categories, and the limited amount of information in a single image. Our architecture outperforms baselines and prior work especially for RGB image inputs.
</details>

### Watch It Move: Unsupervised Discovery of 3D Joints for Re-Posing of Articulated Objects [CVPR 2022]
[📄 Paper](https://arxiv.org/abs/2112.11347) | [🌐 Project Page](https://nvlabs.github.io/watch-it-move/) | [💻 Code](https://github.com/NVlabs/watch-it-move)
- Keywords: Articulation, Unsupervised, MLP
- Input: Multi-view RGB sequence
- Dataset: ZJU-MoCap
<details span>
<summary><b>Abstract</b></summary>
<br>

Rendering articulated objects while controlling their poses is critical to applications such as virtual reality or animation for movies. Manipulating the pose of an object, however, requires the understanding of its underlying structure, that is, its joints and how they interact with each other. Unfortunately, assuming the structure to be known, as existing methods do, precludes the ability to work on new object categories. We propose to learn both the appearance and the structure of previously unseen articulated objects by observing them move from multiple views, with no joints annotation supervision, or information about the structure. We observe that 3D points that are static relative to one another should belong to the same part, and that adjacent parts that move relative to each other must be connected by a joint. To leverage this insight, we model the object parts in 3D as ellipsoids, which allows us to identify joints. We combine this explicit representation with an implicit one that compensates for the approximation introduced. We show that our method works for different structures, from quadrupeds, to single-arm robots, to humans.
</details>

### Understanding 3D Object Articulation in Internet Videos [CVPR 2022]
[📄 Paper](https://arxiv.org/abs/2203.16531) | [🌐 Project Page](https://jasonqsy.github.io/Articulation3D/) | [💻 Code](https://github.com/JasonQSY/Articulation3D)
- Keywords: Articulation, Supervised, FRCNN Net + Optimization
- Input: RGB sequence
- Dataset: Charades
<details span>
<summary><b>Abstract</b></summary>
<br>

We propose to investigate detecting and characterizing the 3D planar articulation of objects from ordinary videos. While seemingly easy for humans, this problem poses many challenges for computers. We propose to approach this problem by combining a top-down detection system that finds planes that can be articulated along with an optimization approach that solves for a 3D plane that can explain a sequence of observed articulations. We show that this system can be trained on a combination of videos and 3D scan datasets. When tested on a dataset of challenging Internet videos and the Charades dataset, our approach obtains strong performance.
</details>

### Towards Real-World Category-level Articulation Pose Estimation [TIP 2022]
[📄 Paper](https://ieeexplore.ieee.org/document/9670684)
- Keywords: Articulation, Supervised, PointNet++
- Input: RGB-D
- Dataset: Multi ReArt-48 (AKB-48)
<details span>
<summary><b>Abstract</b></summary>
<br>

Human life is populated with articulated objects. Current Category-level Articulation Pose Estimation (CAPE) methods are studied under the single-instance setting with a fixed kinematic structure for each category. Considering these limitations, we reform this problem setting for real-world environments and suggest a CAPE-Real (CAPER) task setting. This setting allows varied kinematic structures within a semantic category, and multiple instances to co-exist in an observation of real world. To support this task, we build an articulated model repository ReArt-48 and present an efficient dataset generation pipeline, which contains Fast Articulated Object Modeling (FAOM) and Semi-Authentic MixEd Reality Technique (SAMERT). Accompanying the pipeline, we build a large-scale mixed reality dataset ReArtMix and a real world dataset ReArtVal. We also propose an effective framework ReArtNOCS that exploits RGB-D input to estimate part-level pose for multiple instances in a single forward pass. Extensive experiments demonstrate that the proposed ReArtNOCS can achieve good performance on both CAPER and CAPE settings. We believe it could serve as a strong baseline for future research on the CAPER task.
</details>

### CLA-NeRF: Category-Level Articulated Neural Radiance Field [ICRA 2022]
[📄 Paper](https://arxiv.org/abs/2202.00181)
- Keywords: Articulation, Supervised, NeRF
- Input: Multiple RGB images
- Dataset: PartNet-Mobility
<details span>
<summary><b>Abstract</b></summary>
<br>

We address the task of reconstructing and animating articulated objects from a single RGB image. Our approach, CLA-NeRF, leverages Category-Level Articulated Neural Radiance Fields to achieve this. It is capable of detecting which parts of an object can open or move and determines the motion parameters for each articulated part. This method allows for realistic rendering of objects with articulated parts in novel poses and viewpoints. We demonstrate that CLA-NeRF can handle a variety of object categories, making it versatile for applications in augmented reality, robotics, and beyond.
</details>

### Towards Real-World Category-level Articulation Pose Estimation [TIP 2022]
[📄 Paper](https://ieeexplore.ieee.org/document/9670684)
- Keywords: Articulation, Supervised, PointNet++
- Input: RGB-D
- Dataset: Multi ReArt-48 (AKB-48)
<details span>
<summary><b>Abstract</b></summary>
<br>

We present a new approach for category-level articulation pose estimation (CAPE) in real-world environments. Our proposed task setting, CAPER, allows for varied kinematic structures within a semantic category and multiple instances in a single observation. To facilitate this, we developed ReArt-48, an articulated model repository, and a dataset generation pipeline featuring Fast Articulated Object Modeling (FAOM) and Semi-Authentic MixEd Reality Technique (SAMERT). Our framework, ReArtNOCS, effectively utilizes RGB-D input for estimating part-level pose of multiple instances, showing strong performance on both CAPER and traditional CAPE settings. This work paves the way for more versatile and realistic applications in robotics and computer vision.
</details>

### Understanding 3D Object Articulation in Internet Videos [CVPR 2022]
[📄 Paper](https://arxiv.org/abs/2203.16531) | [🌐 Project Page](https://jasonqsy.github.io/Articulation3D/) | [💻 Code](https://github.com/JasonQSY/Articulation3D)
- Keywords: Articulation, Supervised, FRCNN Net + Optimization
- Input: RGB sequence
- Dataset: Charades
<details span>
<summary><b>Abstract</b></summary>
<br>

We introduce a novel approach to detect and characterize 3D planar articulation of objects from internet videos. By combining a top-down detection system with an optimization strategy, our method finds planes that can be articulated and solves for a 3D plane that explains observed articulations. Trained on a mix of video and 3D scan data, our system demonstrates strong performance on challenging internet videos and the Charades dataset. This work contributes significantly to the understanding of 3D object articulation in dynamic, real-world scenes.
</details>

### OPD: Single-view 3D Openable Part Detection [ECCV 2022]
[📄 Paper](https://arxiv.org/abs/2203.16421) | [🌐 Project Page](https://3dlg-hcvc.github.io/OPD/) | [💻 Code](https://github.com/3dlg-hcvc/OPD)
- Keywords: Articulation, Supervised, MaskRCNN
- Input: Single Point cloud
- Dataset: OPDSynth, OPDReal
<details span>
<summary><b>Abstract</b></summary>
<br>

OPD tackles the challenge of detecting openable parts of objects and their motion parameters from a single view. Using two datasets, OPDSynth and OPDReal, we developed OPDRCNN, a neural network architecture for this purpose. Our work addresses the complexities of this task, especially in generalizing across object categories and the limited information in a single image. Our results demonstrate significant improvements over existing methods, particularly for RGB image inputs, marking a step forward in understanding object articulation.
</details>

### Understanding 3D Object Articulation in Internet Videos
### Towards Real-World Category-level Articulation Pose Estimation
### Self-supervised Neural Articulated Shape and Appearance Models
### Distributional Depth-Based Estimation of Object Articulation Models
### Act the Part: Learning Interaction Strategies for Articulated Object Part Discovery
### Unsupervised Pose-Aware Part Decomposition for 3D Articulated Objects
### ScrewNet: Category-Independent Articulation Model Estimation From Depth Images Using Screw Theory
### CAPTRA: CAtegory-level Pose Tracking for Rigid and Articulated Objects from Point Clouds
### A-SDF: Learning Disentangled Signed Distance Functions for Articulated Shape Representation
### Learning to Infer Kinematic Hierarchies for Novel Object Instances
### SAPIEN: A simulated part-based interactive environment
### Nothing But Geometric Constraints: A Model-Free Method for Articulated Object Pose Estimation
### Category-Level Articulated Object Pose Estimation
### A Hand Motion-guided Articulation and Segmentation Estimation
### RPM-Net: Recurrent Prediction of Motion and Parts from Point Cloud
### Learning to Generalize Kinematic Models to Novel Objects
### Deep Learning Based Robotic Tool Detection and Articulation Estimation With Spatio-Temporal Layers
### Shape2Motion: Joint Analysis of Motion Parts and Attributes from 3D Shapes
### Deep Part Induction from Articulated Object Pairs
### The RBO Dataset of Articulated Objects and Interactions
### Learning to Predict Part Mobility from a Single Static Snapshot
### Reconstructing Articulated Rigged Models from RGB-D Videos
### ShapeNet: An Information-Rich 3D Model Repository
### Towards Understanding Articulated Objects
