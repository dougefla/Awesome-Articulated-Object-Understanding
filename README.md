# Awesome Articulated Objects Understanding
A curated list of resources for articulated objects understanding, including articulation estimation and articulated object reconstruction, excluding human/animal reconstruction. This list is intended to track to the development of articulated objects understanding. If you have any suggestions (missing papers, new tasks, etc.), feel free to pull a request or open an issue.

<details span>
<summary><b>Update log</b></summary>
- 2023/11/28: Initiate with 39 papers.
</details>

## Contents
### PARIS: Part-level Reconstruction and Motion Analysis for Articulated Objects [ICCV 2023, PARIS]
[📄 Paper](https://arxiv.org/abs/2308.07391) | [🌐 Project Page](https://3dlg-hcvc.github.io/paris/) | [💻 Code](https://github.com/3dlg-hcvc/paris)
- Task: Articulation, Reconstruction
- Method: Self-supervised
- Network Keywords: NeRF
- Stages: 1
- Dataset: PartNet-Mobility, MultiScan
- Input: Two sets of multi-view images
<details open>
<summary><b>Abstract</b></summary>
<br>

We design a self-supervised approach without relying any 3D supervision, semantic or motion annotations. The key idea is that we separate the parts from two-state observations by leveraging motion as a cue. Since the motion accounts for the inconsistency between two states, we optimize the motion parameters by registering the moving parts from the input states t to a canonical state t*. During registration, the component that agrees with the transformation is extracted as the movable part. And the one remaining still is extracted as the static part.
</details>

### Ditto in the house: Building articulation models of indoor scenes through interactive perception [ICRA 2023, Ditto in the house]
[📄 Paper](https://arxiv.org/abs/2302.01295) | [🌐 Project Page](https://ut-austin-rpl.github.io/HouseDitto/) | [💻 Code](https://github.com/UT-Austin-RPL/HouseDitto)
- Task: Articulation
- Method: Supervised
- Network Keywords: PointNet++
- Stages: Affordance prediction + Articulation inference
- Dataset: CubiCasa5K
- Input: Two point clouds
<details open>
<summary><b>Abstract</b></summary>
<br>

Our approach, named Ditto in the House, discovers possible articulated objects through affordance prediction, interacts with these objects to produce articulated motions, and infers the articulation properties from the visual observations before and after each interaction. The approach consists of two stages — affordance prediction and articulation inference. During affordance prediction, we pass the static scene point cloud into the affordance network and predict the scene-level affordance map. Then, the robot interacts with the object based on those contact points. During articulation inference, we feed the point cloud observations before and after each interaction into the articulation model network to obtain articulation estimation. By aggregating the estimated articulation models, we build the articulation models of the entire scene.
</details>

### Semi-Weakly Supervised Object Kinematic Motion Prediction [CVPR 2023, SWMP]
[📄 Paper](https://arxiv.org/abs/2303.17774) | [🌐 Project Page](https://vcc.tech/research/2023/SWMP) | [💻 Code](https://github.com/GengxinLiu/SWMP)
- Task: Articulation
- Method: Semi-supervised
- Network Keywords: GNN
- Stages: Prediction + Refinement
- Dataset: PartNet-Mobility, PartNet
- Input: Single Point cloud + segmentation
<details open>
<summary><b>Abstract</b></summary>
<br>

In this paper, we tackle the task of object kinematic motion prediction problem in a semi-weakly supervised manner. Our key observations are two-fold. First, although 3D dataset with fully annotated motion labels is limited, there are existing datasets and methods for object part semantic segmentation at large scale. Second, semantic part segmentation and mobile part segmentation is not always consistent but it is possible to detect the mobile parts from the underlying 3D structure. Towards this end, we propose a graph neural network to learn the map between hierarchical part-level segmentation and mobile parts parameters, which are further refined based on geometric alignment. This network can be first trained on PartNet-Mobility dataset with fully labeled mobility information and then applied on PartNet dataset with fine-grained and hierarchical part-level segmentation. The network predictions yield a large scale of 3D objects with pseudo labeled mobility information and can further be used for weakly-supervised learning with pre-existing segmentation. Our experiments show there are significant performance boosts with the augmented data for previous method designed for kinematic motion prediction on 3D partial scans.
</details>

### CARTO: Category and Joint Agnostic Reconstruction of ARTiculated Objects [CVPR 2023, CARTO]
[📄 Paper](https://arxiv.org/abs/2303.15782) | [🌐 Project Page](http://carto.cs.uni-freiburg.de/) | [💻 Code](https://github.com/robot-learning-freiburg/CARTO)
- Task: Articulation, Reconstruction
- Method: Supervised
- Network Keywords: CenterSnap, SimNet, SDF
- Stages: Encoder-Decoder
- Dataset: PartNet-Mobility
- Input: Single RGB image
<details open>
<summary><b>Abstract</b></summary>
<br>

We present CARTO, a novel approach for reconstructing multiple articulated objects from a single stereo RGB observation. We use implicit object-centric representations and learn a single geometry and articulation decoder for multiple object categories. Despite training on multiple categories, our decoder achieves a comparable reconstruction accuracy to methods that train bespoke decoders separately for each category. Combined with our stereo image encoder we infer the 3D shape, 6D pose, size, joint type, and the joint state of multiple unknown objects in a single forward pass. Our method achieves a 20.4% absolute improvement in mAP 3D IOU50 for novel instances when compared to a two-stage pipeline. Inference time is fast and can run on a NVIDIA TITAN XP GPU at 1 HZ for eight or less objects present. While only trained on simulated data, CARTO transfers to real-world object instances. Code and evaluation data is linked below.
</details>

### CA2T-Net: Category-Agnostic 3D Articulation Transfer from Single Image [Arxiv 2023, CA2T-Net]
[📄 Paper](https://arxiv.org/abs/2301.02232)
- Task: Articulation, Transfer
- Method: Supervised
- Network Keywords: ResNet
- Stages: 1
- Dataset: PartNet-Mobility
- Input: RGB, Mesh
<details open>
<summary><b>Abstract</b></summary>
<br>

We present a neural network approach to transfer the motion from a single image of an articulated object to a rest-state (i.e., unarticulated) 3D model. Our network learns to predict the object's pose, part segmentation, and corresponding motion parameters to reproduce the articulation shown in the input image. The network is composed of three distinct branches that take a shared joint image-shape embedding and is trained end-to-end. Unlike previous methods, our approach is independent of the topology of the object and can work with objects from arbitrary categories. Our method, trained with only synthetic data, can be used to automatically animate a mesh, infer motion from real images, and transfer articulation to functionally similar but geometrically distinct 3D models at test time.
</details>

### Command-driven Articulated Object Understanding and Manipulation [CVPR 2023, Cart]
[📄 Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Chu_Command-Driven_Articulated_Object_Understanding_and_Manipulation_CVPR_2023_paper.html)
- Task: Articulation
- Method: Supervised
- Network Keywords: U-Net
- Stages: Seg-Net + Art-Net
- Dataset: Shape2Motion, PartNet-Mobility
- Input: Single Point cloud
<details open>
<summary><b>Abstract</b></summary>
<br>

We present Cart, a new approach towards articulated-object manipulations by human commands. Beyond the existing work that focuses on inferring articulation structures, we further support manipulating articulated shapes to align them subject to simple command templates. The key of Cart is to utilize the prediction of object structures to connect visual observations with user commands for effective manipulations. It is achieved by encoding command messages for motion prediction and a test-time adaptation to adjust the amount of movement from only command supervision. For a rich variety of object categories, Cart can accurately manipulate object shapes and outperform the state-of-the-art approaches in understanding the inherent articulation structures. Also, it can well generalize to unseen object categories and real-world objects. We hope Cart could open new directions for instructing machines to operate articulated objects.
</details>

### MultiScan: Scalable RGBD scanning for 3D environments with articulated objects [NIPS 2022, MultiScan]
[📄 Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3b3a83a5d86e1d424daefed43d998079-Abstract-Conference.html) | [🌐 Project Page](https://3dlg-hcvc.github.io/multiscan/#/) | [💻 Code](https://github.com/smartscenes/multiscan)
- Task: Dataset
- Dataset: MultiScan
<details open>
<summary><b>Abstract</b></summary>
<br>

We introduce MultiScan, a scalable RGBD dataset construction pipeline leveraging commodity mobile devices to scan indoor scenes with articulated objects and web-based semantic annotation interfaces to efficiently annotate object and part semantics and part mobility parameters. We use this pipeline to collect 273 scans of 117 indoor scenes containing 10957 objects and 5129 parts. The resulting MultiScan dataset provides RGBD streams with per-frame camera poses, textured 3D surface meshes, richly annotated part-level and object-level semantic labels, and part mobility parameters. We validate our dataset on instance segmentation and part mobility estimation tasks and benchmark methods for these tasks from prior work. Our experiments show that part segmentation and mobility estimation in real 3D scenes remain challenging despite recent progress in 3D object segmentation.
</details>

### AKB-48: A real-world articulated object knowledge base [CVPR 2022, AKB-48]
[📄 Paper](https://arxiv.org/abs/2202.08432) | [🌐 Project Page](https://liuliu66.github.io/articulationobjects/)
- Task: Dataset, Articulation, Reconstruct
- Method: Supervised
- Network Keywords: PointNet++, A-SDF
- Stages: 1
- Dataset: AKB-48
- Input: Single RGB-D image
<details open>
<summary><b>Abstract</b></summary>
<br>

Human life is populated with articulated objects. A comprehensive understanding of articulated objects, namely appearance, structure, physics property, and semantics, will benefit many research communities. As current articulated object understanding solutions are usually based on synthetic object dataset with CAD models without physics properties, which prevent satisfied generalization from simulation to real-world applications in visual and robotics tasks. To bridge the gap, we present AKB-48: a large-scale Articulated object Knowledge Base which consists of 2,037 real-world 3D articulated object models of 48 categories. Each object is described by a knowledge graph ArtiKG. To build the AKB-48, we present a fast articulation knowledge modeling (FArM) pipeline, which can fulfill the ArtiKG for an articulated object within 10-15 minutes, and largely reduce the cost for object modeling in the real world. Using our dataset, we propose AKBNet, a novel integral pipeline for Category-level Visual Articulation Manipulation (C-VAM) task, in which we benchmark three sub-tasks, namely pose estimation, object reconstruction and manipulation.
</details>

### FlowBot3D: Learning 3D Articulation Flow to Manipulate Articulated Objects [RSS 2022, FlowBot3D]
[📄 Paper](https://arxiv.org/abs/2205.04382) | [🌐 Project Page](https://sites.google.com/view/articulated-flowbot-3d/home) | [💻 Code](https://github.com/r-pad/flowbot3d)
- Task: Articulation
- Method: Supervised
- Network Keywords: PointNet++
- Stages: 1
- Dataset: PartNet-Mobility
- Input: Single Point cloud
<details open>
<summary><b>Abstract</b></summary>
<br>

We explore a novel method to perceive and manipulate 3D articulated objects that generalizes to enable a robot to articulate unseen classes of objects. We propose a vision-based system that learns to predict the potential motions of the parts of a variety of articulated objects to guide downstream motion planning of the system to articulate the objects. To predict the object motions, we train a neural network to output a dense vector field representing the point-wise motion direction of the points in the point cloud under articulation. We then deploy an analytical motion planner based on this vector field to achieve a policy that yields maximum articulation. We train the vision system entirely in simulation, and we demonstrate the capability of our system to generalize to unseen object instances and novel categories in both simulation and the real world, deploying our policy on a Sawyer robot with no finetuning. Results show that our system achieves state-of-the-art performance in both simulated and real-world experiments.
</details>

### Category-Independent Articulated Object Tracking with Factor Graphs [IROS 2022, Heppert etc.]
[📄 Paper](https://arxiv.org/abs/2205.03721) | [🌐 Project Page](https://sites.google.com/view/category-independentarticulate/category-independent-articulated-object-tracking-with-factor-graphs) | [💻 Code](https://github.com/SuperN1ck/cat-ind-fg)
- Task: Articulation
- Method: Supervised+Factor Graph
- Stages: Part Pose Tracking + Articualtion
- Dataset: PartNet-Mobility
- Input: RGB-D sequence
<details open>
<summary><b>Abstract</b></summary>
<br>

Robots deployed in human-centric environments may need to manipulate a diverse range of articulated objects, such as doors, dishwashers, and cabinets. Articulated objects often come with unexpected articulation mechanisms that are inconsistent with categorical priors: for example, a drawer might rotate about a hinge joint instead of sliding open. We propose a category-independent framework for predicting the articulation models of unknown objects from sequences of RGB-D images. The prediction is performed by a two-step process: first, a visual perception module tracks object part poses from raw images, and second, a factor graph takes these poses and infers the articulation model including the current configuration between the parts as a 6D twist. We also propose a manipulation-oriented metric to evaluate predicted joint twists in terms of how well a compliant robot controller would be able to manipulate the articulated object given the predicted twist. We demonstrate that our visual perception and factor graph modules outperform baselines on simulated data and show the applicability of our factor graph on real world data.
</details>

### Ditto: Building Digital Twins of Articulated Objects from Interaction [CVPR 2022, Ditto]
[📄 Paper](https://arxiv.org/abs/2202.08227) | [🌐 Project Page](https://rpl.cs.utexas.edu/publications/2022/06/19/jiang-cvpr22-ditto/) | [💻 Code](https://github.com/UT-Austin-RPL/Ditto)
- Task: Articulation, Tracking
- Method: Supervised
- Network Keywords: PointNet++, NeRF
- Stages: 1
- Dataset: Shape2Motion
- Input: Two point clouds
<details open>
<summary><b>Abstract</b></summary>
<br>

Digitizing physical objects into the virtual world has the potential to unlock new research and applications in embodied AI and mixed reality. This work focuses on recreating interactive digital twins of real-world articulated objects, which can be directly imported into virtual environments. We introduce Ditto to learn articulation model estimation and 3D geometry reconstruction of an articulated object through interactive perception. Given a pair of visual observations of an articulated object before and after interaction, Ditto reconstructs part-level geometry and estimates the articulation model of the object. We employ implicit neural representations for joint geometry and articulation modeling. Our experiments show that Ditto effectively builds digital twins of articulated objects in a category-agnostic way. We also apply Ditto to real-world objects and deploy the recreated digital twins in physical simulation.
</details>

### OPD: Single-view 3D Openable Part Detection [ECCV 2022, OPD]
[📄 Paper](https://arxiv.org/abs/2203.16421) | [🌐 Project Page](https://3dlg-hcvc.github.io/OPD/) | [💻 Code](https://github.com/3dlg-hcvc/OPD)
- Task: Articulation
- Method: Supervised
- Network Keywords: MaskRCNN
- Stages: 1
- Dataset: OPDSynth, OPDReal
- Input: Single Point cloud
<details open>
<summary><b>Abstract</b></summary>
<br>

We address the task of predicting what parts of an object can open and how they move when they do so. The input is a single image of an object, and as output we detect what parts of the object can open, and the motion parameters describ- ing the articulation of each openable part. To tackle this task, we create two datasets of 3D objects: OPDSynth based on existing synthetic objects, and OPDReal based on RGBD reconstructions of real objects. We then design OPDRCNN, a neural architecture that detects openable parts and predicts their motion parameters. Our experiments show that this is a challenging task especially when considering general- ization across object categories, and the limited amount of information in a single image. Our architecture outperforms baselines and prior work especially for RGB image inputs.
</details>

### CLA-NeRF: Category-Level Articulated Neural Radiance Field [ICRA 2022, CLA-NeRF]
[📄 Paper](https://arxiv.org/abs/2202.00181)
- Task: Articulation
- Method: Supervised
- Network Keywords: NeRF
- Stages: 1
- Dataset: PartNet-Mobility
- Input: Multiple RGB images
<details open>
<summary><b>Abstract</b></summary>
<br>

We address the task of predicting what parts of an object can open and how they move when they do so. The input is a single image of an object, and as output we detect what parts of the object can open, and the motion parameters describing the articulation of each openable part. To tackle this task, we create two datasets of 3D objects: OPDSynth based on existing synthetic objects, and OPDReal based on RGBD reconstructions of real objects. We then design OPDRCNN, a neural architecture that detects openable parts and predicts their motion parameters. Our experiments show that this is a challenging task especially when considering generalization across object categories, and the limited amount of information in a single image. Our architecture outperforms baselines and prior work especially for RGB image inputs.
</details>

### Watch It Move: Unsupervised Discovery of 3D Joints for Re-Posing of Articulated Objects [CVPR 2022, Watch It Move]
[📄 Paper](https://arxiv.org/abs/2112.11347) | [🌐 Project Page](https://nvlabs.github.io/watch-it-move/) | [💻 Code](https://github.com/NVlabs/watch-it-move)
- Task: Articulation
- Method: Unsupervised
- Network Keywords: MLP
- Dataset: ZJU-MoCap
- Input: Multi-view RGB sequence
<details open>
<summary><b>Abstract</b></summary>
<br>

Rendering articulated objects while controlling their poses is critical to applications such as virtual reality or animation for movies. Manipulating the pose of an object, however, requires the understanding of its underlying structure, that is, its joints and how they interact with each other. Unfortunately, assuming the structure to be known, as existing methods do, precludes the ability to work on new object categories. We propose to learn both the appearance and the structure of previously unseen articulated objects by observing them move from multiple views, with no joints annotation supervision, or information about the structure. We observe that 3D points that are static relative to one another should belong to the same part, and that adjacent parts that move relative to each other must be connected by a joint. To leverage this insight, we model the object parts in 3D as ellipsoids, which allows us to identify joints. We combine this explicit representation with an implicit one that compensates for the approximation introduced. We show that our method works for different structures, from quadrupeds, to single-arm robots, to humans.
</details>

### Understanding 3D Object Articulation in Internet Videos [CVPR 2022, 3DADN]
[📄 Paper](https://arxiv.org/abs/2203.16531) | [🌐 Project Page](https://jasonqsy.github.io/Articulation3D/) | [💻 Code](https://github.com/JasonQSY/Articulation3D)
- Task: Articulation
- Method: Supervised
- Network Keywords: FRCNN
- Stages: Net + Optimization
- Dataset: Charades
- Input: RGB sequence
<details open>
<summary><b>Abstract</b></summary>
<br>

We propose to investigate detecting and characterizing the 3D planar articulation of objects from ordinary videos. While seemingly easy for humans, this problem poses many challenges for computers. We propose to approach this problem by combining a top-down detection system that finds planes that can be articulated along with an optimization approach that solves for a 3D plane that can explain a sequence of observed articulations. We show that this system can be trained on a combination of videos and 3D scan datasets. When tested on a dataset of challenging Internet videos and the Charades dataset, our approach obtains strong performance. 
</details>

### Towards Real-World Category-level Articulation Pose Estimation [TIP 2022, CAPER]
[📄 Paper](https://ieeexplore.ieee.org/document/9670684)
- Task: Articulation
- Method: Supervised
- Network Keywords: PointNet++
- Stages: Multiple Stages
- Dataset: ReArt-48 (AKB-48)
- Input: RGB-D
<details open>
<summary><b>Abstract</b></summary>
<br>

Human life is populated with articulated objects. Current Category-level Articulation Pose Estimation (CAPE) methods are studied under the single-instance setting with a fixed kinematic structure for each category. Considering these limitations, we reform this problem setting for real-world environments and suggest a CAPE-Real (CAPER) task setting. This setting allows varied kinematic structures within a semantic category, and multiple instances to co-exist in an observation of real world. To support this task, we build an articulated model repository ReArt-48 and present an efficient dataset generation pipeline, which contains Fast Articulated Object Modeling (FAOM) and Semi-Authentic MixEd Reality Technique (SAMERT). Accompanying the pipeline, we build a large-scale mixed reality dataset ReArtMix and a real world dataset ReArtVal. We also propose an effective framework ReArtNOCS that exploits RGB-D input to estimate part-level pose for multiple instances in a single forward pass. Extensive experiments demonstrate that the proposed ReArtNOCS can achieve good performance on both CAPER and CAPE settings. We believe it could serve as a strong baseline for future research on the CAPER task.
</details>

### Self-supervised Neural Articulated Shape and Appearance Models [CVPR 2022, NASAM]
[📄 Paper](https://arxiv.org/abs/2205.08525) | [🌐 Project Page](https://weify627.github.io/nasam/)
- Task: Articulation, Reconstruction
- Method: Self-supervised
- Network Keywords: IDR
- Stages: 1
- Dataset: PartNet-Mobility
- Input: Shape code
<details open>
<summary><b>Abstract</b></summary>
<br>

Learning geometry, motion, and appearance priors of object classes is important for the solution of a large variety of computer vision problems. While the majority of approaches has focused on static objects, dynamic objects, especially with controllable articulation, are less explored. We propose a novel approach for learning a representation of the geometry, appearance, and motion of a class of articulated objects given only a set of color images as input. In a self-supervised manner, our novel representation learns shape, appearance, and articulation codes that enable independent control of these semantic dimensions. Our model is trained end-to-end without requiring any articulation annotations. Experiments show that our approach performs well for different joint types, such as revolute and prismatic joints, as well as different combinations of these joints. Compared to state of the art that uses direct 3D supervision and does not output appearance, we recover more faithful geometry and appearance from 2D observations only. In addition, our representation enables a large variety of applications, such as few-shot reconstruction, the generation of novel articulations, and novel view-synthesis.
</details>

### Distributional Depth-Based Estimation of Object Articulation Models [CoRL 2021, DUST-Net]
[📄 Paper](https://arxiv.org/abs/2108.05875v1) | [🌐 Project Page](https://pearl-utexas.github.io/DUST-net/) | [💻 Code](https://github.com/Pearl-UTexas/DUST-net)
- Task: Articulation
- Method: Supervised
- Network Keywords: DUST-NET
- Stages: 1
- Dataset: PartNet-Mobility, SyntheticArticulatedData
- Input: Depth sequence
<details open>
<summary><b>Abstract</b></summary>
<br>

We propose a method that efficiently learns distributions over articulation model parameters directly from depth images without the need to know articulation model categories a priori. By contrast, existing methods that learn articulation models from raw observations typically only predict point estimates of the model parameters, which are insufficient to guarantee the safe manipulation of articulated objects. Our core contributions include a novel representation for distributions over rigid body transformations and articulation model parameters based on screw theory, von Mises-Fisher distributions, and Stiefel manifolds. Combining these concepts allows for an efficient, mathematically sound representation that implicitly satisfies the constraints that rigid body transformations and articulations must adhere to. Leveraging this representation, we introduce a novel deep learning based approach, DUST-net, that performs category-independent articulation model estimation while also providing model uncertainties. We evaluate our approach on several benchmarking datasets and real-world objects and compare its performance with two current state-of-the-art methods. Our results demonstrate that DUST-net can successfully learn distributions over articulation models for novel objects across articulation model categories, which generate point estimates with better accuracy than state-of-the-art methods and effectively capture the uncertainty over predicted model parameters due to noisy inputs.
</details>

### Act the Part: Learning Interaction Strategies for Articulated Object Part Discovery [ICCV 2021, AtP]
[📄 Paper](https://arxiv.org/abs/2105.01047)
- Task: Articulation
- Network Keywords: RL
- Dataset: Partnet-Mobility
- Input: RGB sequence
<details open>
<summary><b>Abstract</b></summary>
<br>

People often use physical intuition when manipulating articulated objects, irrespective of object semantics. Motivated by this observation, we identify an important embodied task where an agent must play with objects to recover their parts. To this end, we introduce Act the Part (AtP) to learn how to interact with articulated objects to discover and segment their pieces. By coupling action selection and motion segmentation, AtP is able to isolate structures to make perceptual part recovery possible without semantic labels. Our experiments show AtP learns efficient strategies for part discovery, can generalize to unseen categories, and is capable of conditional reasoning for the task. Although trained in simulation, we show convincing transfer to real world data with no fine-tuning.
</details>

### Unsupervised Pose-Aware Part Decomposition for 3D Articulated Objects [ECCV 2021, PPD]
[📄 Paper](https://arxiv.org/abs/2110.04411)
- Task: Articulation
- Method: Unsupervised
- Network Keywords: PointNet++, INR
- Stages: Multiple Stages
- Dataset: Shape2motion, Partnet-Mobility
- Input: Single Point cloud
<details open>
<summary><b>Abstract</b></summary>
<br>

Articulated objects exist widely in the real world. However, previous 3D generative methods for unsupervised part decomposition are unsuitable for such objects, because they assume a spatially fixed part location, resulting in inconsistent part parsing. In this paper, we propose PPD (unsupervised Pose-aware Part Decomposition) to address a novel setting that explicitly targets man-made articulated objects with mechanical joints, considering the part poses. We show that category-common prior learning for both part shapes and poses facilitates the unsupervised learning of (1) part decomposition with non-primitive-based implicit representation, and (2) part pose as joint parameters under single-frame shape supervision. We evaluate our method on synthetic and real datasets, and we show that it outperforms previous works in consistent part parsing of the articulated objects based on comparable part pose estimation performance to the supervised baseline
</details>

### ScrewNet: Category-Independent Articulation Model Estimation From Depth Images Using Screw Theory [ICRA 2021, ScrewNet]
[📄 Paper](https://arxiv.org/abs/2008.10518) | [🌐 Project Page](https://pearl-utexas.github.io/ScrewNet/) | [💻 Code](https://github.com/Pearl-UTexas/ScrewNet)
- Task: Articulation
- Method: Supervised
- Network Keywords: ResNet+LSTM
- Stages: 1
- Dataset: PartNet-Mobility
- Input: Depth sequence
<details open>
<summary><b>Abstract</b></summary>
<br>

Robots in human environments will need to interact with a wide variety of articulated objects such as cabinets, drawers, and dishwashers while assisting humans in performing day-to-day tasks. Existing methods either require objects to be textured or need to know the articulation model category a priori for estimating the model parameters for an articulated object. We propose ScrewNet, a novel approach that estimates an object’s articulation model directly from depth images without requiring a priori knowledge of the articulation model category. ScrewNet uses screw theory to unify the representation of different articulation types and perform category-independent articulation model estimation. We evaluate our approach on two benchmarking datasets and compare its performance with a current state-of-the-art method. Results demonstrate that ScrewNet can successfully estimate the articulation models and their parameters for novel objects across articulation model categories with better on average accuracy than the prior state-of-the-art method.
</details>

### CAPTRA: CAtegory-level Pose Tracking for Rigid and Articulated Objects from Point Clouds [ICCV 2021, CAPTRA]
[📄 Paper](https://arxiv.org/abs/2104.03437) | [🌐 Project Page](https://yijiaweng.github.io/CAPTRA/) | [💻 Code](https://github.com/halfsummer11/CAPTRA)
- Task: Articulation, Pose Tracking, Dataset
- Method: Supervised
- Network Keywords: PointNet++
- Stages: 1
- Dataset: NOCS-REAL275, PartNet-Mobility, BMVC
- Input: Depth sequence
<details open>
<summary><b>Abstract</b></summary>
<br>

In this work, we tackle the problem of category-level online pose tracking of objects from point cloud sequences. For the first time, we propose a unified framework that can handle 9DoF pose tracking for novel rigid object instances as well as per-part pose tracking for articulated objects from known categories. Here the 9DoF pose, comprising 6D pose and 3D size, is equivalent to a 3D amodal bounding box representation with free 6D pose. Given the depth point cloud at the current frame and the estimated pose from the last frame, our novel end-to-end pipeline learns to accurately update the pose. Our pipeline is composed of three modules: 1) a pose canonicalization module that normalizes the pose of the input depth point cloud; 2) RotationNet, a module that directly regresses small interframe delta rotations; and 3) CoordinateNet, a module that predicts the normalized coordinates and segmentation, enabling analytical computation of the 3D size and translation. Leveraging the small pose regime in the pose-canonicalized point clouds, our method integrates the best of both worlds by combining dense coordinate prediction and direct rotation regression, thus yielding an end-to-end differentiable pipeline optimized for 9DoF pose accuracy (without using non-differentiable RANSAC). Our extensive experiments demonstrate that our method achieves new state-of-the-art performance on category-level rigid object pose (NOCS-REAL275) and articulated object pose benchmarks (SAPIEN, BMVC) at the fastest FPS ~12.
</details>

### A-SDF: Learning Disentangled Signed Distance Functions for Articulated Shape Representation [ICCV 2021, A-SDF]
[📄 Paper](https://arxiv.org/abs/2104.07645) | [🌐 Project Page](https://jitengmu.github.io/A-SDF/) | [💻 Code](https://github.com/JitengMu/A-SDF)
- Task: Articulation, Reconstruction
- Method: Supervised
- Network Keywords: SDF
- Stages: 1
- Dataset: Shape2Motion
- Input: Single Depth Image
<details open>
<summary><b>Abstract</b></summary>
<br>

Recent work has made significant progress on using implicit functions, as a continuous representation for 3D rigid object shape reconstruction. However, much less effort has been devoted to modeling general articulated objects. Compared to rigid objects, articulated objects have higher degrees of freedom, which makes it hard to generalize to unseen shapes. To deal with the large shape variance, we introduce Articulated Signed Distance Functions (A-SDF) to represent articulated shapes with a disentangled latent space, where we have separate codes for encoding shape and articulation. With this disentangled continuous representation, we demonstrate that we can control the articulation input and animate unseen instances with unseen joint angles. Furthermore, we propose a Test-Time Adaptation inference algorithm to adjust our model during inference. We demonstrate our model generalize well to out-of-distribution and unseen data, e.g., partial point clouds and real-world depth images.
</details>

### Learning to Infer Kinematic Hierarchies for Novel Object Instances [ICRA 2021, Abdul-Rashid etc.]
[📄 Paper](https://arxiv.org/abs/2110.07911)
- Task: Articulation
- Method: Supervised
- Network Keywords: PointNet++, GNN
- Stages: 2
- Dataset: PartNet-Mobility
- Input: Single Point cloud
<details open>
<summary><b>Abstract</b></summary>
<br>

Manipulating an articulated object requires perceiving itskinematic hierarchy: its parts, how each can move, and howthose motions are coupled. Previous work has explored per-ception for kinematics, but none infers a complete kinematichierarchy on never-before-seen object instances, without relyingon a schema or template. We present a novel perception systemthat achieves this goal. Our system infers the moving parts ofan object and the kinematic couplings that relate them. Toinfer parts, it uses a point cloud instance segmentation neuralnetwork and to infer kinematic hierarchies, it uses a graphneural network to predict the existence, direction, and typeof edges (i.e. joints) that relate the inferred parts. We trainthese networks using simulated scans of synthetic 3D models.We evaluate our system on simulated scans of 3D objects, andwe demonstrate a proof-of-concept use of our system to drivereal-world robotic manipulation.
</details>

### SAPIEN: A simulated part-based interactive environment [CVPR 2020, SPAIEN]
[📄 Paper](https://arxiv.org/abs/2003.08515) | [🌐 Project Page](https://sapien.ucsd.edu/) | [💻 Code](https://github.com/haosulab/SAPIEN)
- Task: Articulation, Dataset
- Dataset: PartNet-Mobility
<details open>
<summary><b>Abstract</b></summary>
<br>

Building home assistant robots has long been a pursuit for vision and robotics researchers. To achieve this task, a simulated environment with physically realistic simulation, sufficient articulated objects, and transferability to the real robot is indispensable. Existing environments achieve these requirements for robotics simulation with different levels of simplification and focus. We take one step further in constructing an environment that supports household tasks for training robot learning algorithm. Our work, SAPIEN, is a realistic and physics-rich simulated environment that hosts a large-scale set for articulated objects. Our SAPIEN enables various robotic vision and interaction tasks that require detailed part-level understanding.We evaluate state-of-the-art vision algorithms for part detection and motion attribute recognition as well as demonstrate robotic interaction tasks using heuristic approaches and reinforcement learning algorithms. We hope that our SAPIEN can open a lot of research directions yet to be explored, including learning cognition through interaction, part motion discovery, and construction of robotics-ready simulated game environment.
</details>

### Nothing But Geometric Constraints: A Model-Free Method for Articulated Object Pose Estimation [Arxiv 2020, GC-Pose]
[📄 Paper](https://arxiv.org/abs/2012.00088)
- Task: Articulation
- Method: Unsupervised
- Network Keywords: FlowNet2
- Stages: 2
- Dataset: PartNet-Mobility, UR5, synthetic
- Input: RGB sequence
<details open>
<summary><b>Abstract</b></summary>
<br>

We propose an unsupervised vision-based system to estimate the joint configurations of the robot arm from a sequence of RGB or RGB-D images without knowing the model a priori, and then adapt it to the task of category-independent articulated object pose estimation. We combine a classical geometric formulation with deep learning and extend the use of epipolar constraint to multi-rigid-body systems to solve this task. Given a video sequence, the optical flow is estimated to get the pixel-wise dense correspondences. After that, the 6D pose is computed by a modified PnP algorithm. The key idea is to leverage the geometric constraints and the constraint between multiple frames. Furthermore, we build a synthetic dataset with different kinds of robots and multi-joint articulated objects for the research of vision-based robot control and robotic vision. We demonstrate the effectiveness of our method on three benchmark datasets and show that our method achieves higher accuracy than the state-of-the-art supervised methods in estimating joint angles of robot arms and articulated objects.
</details>

### Category-Level Articulated Object Pose Estimation [CVPR 2020, ANCSH]
[📄 Paper](https://arxiv.org/abs/1912.11913) | [🌐 Project Page](https://articulated-pose.github.io/) | [💻 Code](https://github.com/dragonlong/articulated-pose)
- Task: Articulation
- Method: Supervised
- Network Keywords: PointNet++
- Stages: 2
- Dataset: Shape2Motion, PartNet-Mobility, 
- Input: Single Point cloud
<details open>
<summary><b>Abstract</b></summary>
<br>

This project addresses the task of category-level pose estimation for articulated objects from a single depth image. We present a novel category-level approach that correctly accommodates object instances previously unseen during training. We introduce Articulation-aware Normalized Coordinate Space Hierarchy (ANCSH) - a canonical representation for different articulated objects in a given category. As the key to achieve intra-category generalization, the representation constructs a canonical object space as well as a set of canonical part spaces. The canonical object space normalizes the object orientation,scales and articulations (e.g. joint parameters and states) while each canonical part space further normalizes its part pose and scale. We develop a deep network based on PointNet++ that predicts ANCSH from a single depth point cloud, including part segmentation, normalized coordinates, and joint parameters in the canonical object space. By leveraging the canonicalized joints, we demonstrate: 1) improved performance in part pose and scale estimations using the induced kinematic constraints from joints; 2) high accuracy for joint parameter estimation in camera space.
</details>

### A Hand Motion-guided Articulation and Segmentation Estimation [ROMAN 2020, Hartanto etc.]
[📄 Paper](https://arxiv.org/abs/2005.03691)
- Task: Articulation
- Method: Unsupervised
- Input: RGB-D sequence
<details open>
<summary><b>Abstract</b></summary>
<br>

In this paper, we present a method for simultaneous articulation model estimation and segmentation of an articulated object in RGB-D images using human hand motion. Our method uses the hand motion in the processes of the initial articulation model estimation, ICP-based model parameter optimization, and region selection of the target object. The hand motion gives an initial guess of the articulation model: prismatic or revolute joint. The method estimates the joint parameters by aligning the RGB-D images with the constraint of the hand motion. Finally, the target regions are selected from the cluster regions which move symmetrically along with the articulation model. Our experimental results show the robustness of the proposed method for the various objects.
</details>

### RPM-Net: Recurrent Prediction of Motion and Parts from Point Cloud [ToG 2019, RPM-Net]
[📄 Paper](https://dl.acm.org/doi/10.1145/3355089.3356573) | [💻 Code](https://github.com/Salingo/RPM-Net)
- Task: Articulation
- Method: Supervised
- Network Keywords: RNN
- Stages: 1
- Input: Single Point cloud
<details open>
<summary><b>Abstract</b></summary>
<br>

We introduce RPM-Net, a deep learning-based approach which simultaneously infers movable parts and hallucinates their motions from a single, un-segmented, and possibly partial, 3D point cloud shape. RPM-Net is a novel Recurrent Neural Network (RNN), composed of an encoder-decoder pair with interleaved Long Short-Term Memory (LSTM) components, which together predict a temporal sequence of pointwise displacements for the input point cloud. At the same time, the displacements allow the network to learn movable parts, resulting in a motion-based shape segmentation. Recursive applications of RPM-Net on the obtained parts can predict finer-level part motions, resulting in a hierarchical object segmentation. Furthermore, we develop a separate network to estimate part mobilities, e.g., per-part motion parameters, from the segmented motion sequence. Both networks learn deep predictive models from a training set that exemplifies a variety of mobilities for diverse objects. We show results of simultaneous motion and part predictions from synthetic and real scans of 3D objects exhibiting a variety of part mobilities, possibly involving multiple movable parts.
</details>

### Learning to Generalize Kinematic Models to Novel Objects [CoRL 2019, Abbatematteo etc.]
[📄 Paper](https://proceedings.mlr.press/v100/abbatematteo20a.html) | [💻 Code](https://github.com/babbatem/SyntheticArticulatedData)
- Task: Articulation
- Method: Supervised
- Network Keywords: Mixture density networks
- Stages: 1
- Dataset: SyntheticArticulatedData
- Input: Single Point cloud
<details open>
<summary><b>Abstract</b></summary>
<br>

Robots operating in human environments must be capable of interacting with a wide variety of articulated objects such as cabinets, refrigerators, and drawers. Existing approaches require human demonstration or minutes of interaction to fit kinematic models to each novel object from scratch. We present a framework for estimating the kinematic model and configuration of previously unseen articulated objects, conditioned upon object type, from as little as a single observation. We train our system in simulation with a novel dataset of synthetic articulated objects; at runtime, our model can predict the shape and kinematic model of an object from depth sensor data. We demonstrate that our approach enables a MOVO robot to view an object with its RGB-D sensor, estimate its motion model, and use that estimate to interact with the object.
</details>

### Deep Learning Based Robotic Tool Detection and Articulation Estimation With Spatio-Temporal Layers [RA-L 2019, Colleoni etc.]
[📄 Paper](https://ieeexplore.ieee.org/document/8715379)
- Task: Articulation
- Method: Supervised
- Network Keywords: 3D FCNN
- Stages: 1
- Dataset: EndoVis, UCL dVRK
- Input: RGB sequence
<details open>
<summary><b>Abstract</b></summary>
<br>

Surgical-tool joint detection from laparoscopic images is an important but challenging task in computer-assisted minimally invasive surgery. Illumination levels, variations in background and the different number of tools in the field of view, all pose difficulties to algorithm and model training. Yet, such challenges could be potentially tackled by exploiting the temporal information in laparoscopic videos to avoid per frame handling of the problem. In this letter, we propose a novel encoder-decoder architecture for surgical instrument joint detection and localization that uses three-dimensional convolutional layers to exploit spatio-temporal features from laparoscopic videos. When tested on benchmark and custom-built datasets, a median Dice similarity coefficient of 85.1% with an interquartile range of 4.6% highlights performance better than the state of the art based on single-frame processing. Alongside novelty of the network architecture, the idea for inclusion of temporal information appears to be particularly useful when processing images with unseen backgrounds during the training phase, which indicates that spatio-temporal features for joint detection help to generalize the solution.
</details>

### Shape2Motion: Joint Analysis of Motion Parts and Attributes from 3D Shapes [CVPR 2019, Shape2Motion]
[📄 Paper](https://arxiv.org/abs/1903.03911) | [💻 Code](https://github.com/wangxiaogang866/Shape2Motion)
- Task: Articulation, Dataset
- Method: Supervised
- Network Keywords: PointNet++
- Stages: Motion Part+Motion Attribute+Proposal Matching
- Dataset: Shape2Motion
- Input: Single Point cloud
<details open>
<summary><b>Abstract</b></summary>
<br>

For the task of mobility analysis of 3D shapes, we propose joint analysis for simultaneous motion part segmentation and motion attribute estimation, taking a single 3D model as input. The problem is significantly different from those tackled in the existing works which assume the availability of either a pre-existing shape segmentation or multiple 3D models in different motion states. To that end, we develop Shape2Motion which takes a single 3D point cloud as input, and jointly computes a mobility-oriented segmentation and the associated motion attributes. Shape2Motion is comprised of two deep neural networks designed for mobility proposal generation and mobility optimization, respectively. The key contribution of these networks is the novel motion-driven features and losses used in both motion part segmentation and motion attribute estimation. This is based on the observation that the movement of a functional part preserves the shape structure. We evaluate Shape2Motion with a newly proposed benchmark for mobility analysis of 3D shapes. Results demonstrate that our method achieves the state-of-the-art performance both in terms of motion part segmentation and motion attribute estimation.
</details>

### Deep Part Induction from Articulated Object Pairs [SIGGRAPH Asia 2018, Yi etc.]
[📄 Paper](https://arxiv.org/abs/1809.07417) | [💻 Code](https://github.com/ericyi/articulated-part-induction)
- Task: Articulation
- Method: Supervised
- Network Keywords: PointNet++
- Stages: Correspondence proposal+Flow+Segmentation
- Dataset: ShapePFCN
- Input: Point Cloud + CAD model
<details open>
<summary><b>Abstract</b></summary>
<br>

Object functionality is often expressed through part articulation -- as when the two rigid parts of a scissor pivot against each other to perform the cutting function. Such articulations are often similar across objects within the same functional category. In this paper, we explore how the observation of different articulation states provides evidence for part structure and motion of 3D objects. Our method takes as input a pair of unsegmented shapes representing two different articulation states of two functionally related objects, and induces their common parts along with their underlying rigid motion. This is a challenging setting, as we assume no prior shape structure, no prior shape category information, no consistent shape orientation, the articulation states may belong to objects of different geometry, plus we allow inputs to be noisy and partial scans, or point clouds lifted from RGB images. Our method learns a neural network architecture with three modules that respectively propose correspondences, estimate 3D deformation flows, and perform segmentation. To achieve optimal performance, our architecture alternates between correspondence, deformation flow, and segmentation prediction iteratively in an ICP-like fashion. Our results demonstrate that our method significantly outperforms state-of-the-art techniques in the task of discovering articulated parts of objects. In addition, our part induction is object-class agnostic and successfully generalizes to new and unseen objects.
</details>

### The RBO Dataset of Articulated Objects and Interactions [IJRR 2019, RBO]
[📄 Paper](https://arxiv.org/abs/1806.06465) | [🌐 Project Page](https://tu-rbo.github.io/articulated-objects/)
- Task: Articulation
<details open>
<summary><b>Abstract</b></summary>
<br>

We present a dataset with models of 14 articulated objects commonly found in human environments and with RGB-D video sequences and wrenches recorded of human interactions with them. The 358 interaction sequences total 67 minutes of human manipulation under varying experimental conditions (type of interaction, lighting, perspective, and background). Each interaction with an object is annotated with the ground truth poses of its rigid parts and the kinematic state obtained by a motion capture system. For a subset of 78 sequences (25 minutes), we also measured the interaction wrenches. The object models contain textured three-dimensional triangle meshes of each link and their motion constraints. We provide Python scripts to download and visualize the data.
</details>

### Learning to Predict Part Mobility from a Single Static Snapshot [TOG 2017, Hu etc.]
[📄 Paper](https://dl.acm.org/doi/10.1145/3130800.3130811)
- Task: Articulation
- Method: Supervised
- Stages: S-D mapping + Motion prediction
- Dataset: ShapeNetCore
- Input: Snapshots
<details open>
<summary><b>Abstract</b></summary>
<br>

We introduce a method for learning a model for the mobility of parts in 3D objects. Our method allows not only to understand the dynamic functionalities of one or more parts in a 3D object, but also to apply the mobility functions to static 3D models. Specifically, the learned part mobility model can predict mobilities for parts of a 3D object given in the form of a single static snapshot reflecting the spatial configuration of the object parts in 3D space, and transfer the mobility from relevant units in the training data. The training data consists of a set of mobility units of different motion types. Each unit is composed of a pair of 3D object parts (one moving and one reference part), along with usage examples consisting of a few snapshots capturing different motion states of the unit. Taking advantage of a linearity characteristic exhibited by most part motions in everyday objects, and utilizing a set of part-relation descriptors, we define a mapping from static snapshots to dynamic units. This mapping employs a motion-dependent snapshot-to-unit distance obtained via metric learning. We show that our learning scheme leads to accurate motion prediction from single static snapshots and allows proper motion transfer. We also demonstrate other applications such as motion-driven object detection and motion hierarchy construction.
</details>

### 3D Shape Segmentation with Projective Convolutional Networks [CVPR 2017, ShapePFCN]
[📄 Paper](https://arxiv.org/abs/1612.02808) | [🌐 Project Page](https://people.cs.umass.edu/~kalo/papers/shapepfcn/index.html) | [💻 Code](https://github.com/kalov/ShapePFCN)
- Task: Dataset, Part Segmentation
- Dataset: ShapePFCN
<details open>
<summary><b>Abstract</b></summary>
<br>

This paper introduces a deep architecture for segmenting 3D objects into their labeled semantic parts. Our architecture combines image-based Fully Convolutional Networks (FCNs) and surface-based Conditional Random Fields (CRFs) to yield coherent segmentations of 3D shapes. The image-based FCNs are used for efficient view-based reasoning about 3D object parts. Through a special projection layer, FCN outputs are effectively aggregated across multiple views and scales, then are projected onto the 3D object surfaces. Finally, a surface-based CRF combines the projected outputs with geometric consistency cues to yield coherent segmentations. The whole architecture (multi-view FCNs and CRF) is trained end-to-end. Our approach significantly outperforms the existing state-of-the-art methods in the currently largest segmentation benchmark (ShapeNet). Finally, we demonstrate promising segmentation results on noisy 3D shapes acquired from consumer-grade depth cameras.
</details>

### Reconstructing Articulated Rigged Models from RGB-D Videos [ECCVW 2016, Tzionas etc.]
[📄 Paper](https://arxiv.org/abs/1609.01371)
- Task: Articulation, Reconstruction
- Method: Optimization
- Stages: Mesh Tracking + Skeleton Reconstruction
- Input: Depth sequence + CAD model
<details open>
<summary><b>Abstract</b></summary>
<br>

Although commercial and open-source software exist to reconstruct a static object from a sequence recorded with an RGB-D sensor, there is a lack of tools that build rigged models of articulated objects that deform realistically and can be used for tracking or animation. In this work, we fill this gap and propose a method that creates a fully rigged model of an articulated object from depth data of a single sensor. To this end, we combine deformable mesh tracking, motion segmentation based on spectral clustering and skeletonization based on mean curvature flow. The fully rigged model then consists of a watertight mesh, embedded skeleton, and skinning weights.
</details>

### ShapeNet: An Information-Rich 3D Model Repository [Arxiv 2015, ShapeNet]
[📄 Paper](https://arxiv.org/abs/1512.03012) | [🌐 Project Page](https://shapenet.org/)
- Task: Dataset
- Dataset: ShapeNet
<details open>
<summary><b>Abstract</b></summary>
<br>

We present ShapeNet: a richly-annotated, large-scale repository of shapes represented by 3D CAD models of objects. ShapeNet contains 3D models from a multitude of semantic categories and organizes them under the WordNet taxonomy. It is a collection of datasets providing many semantic annotations for each 3D model such as consistent rigid alignments, parts and bilateral symmetry planes, physical sizes, keywords, as well as other planned annotations. Annotations are made available through a public web-based interface to enable data visualization of object attributes, promote data-driven geometric analysis, and provide a large-scale quantitative benchmark for research in computer graphics and vision. At the time of this technical report, ShapeNet has indexed more than 3,000,000 models, 220,000 models out of which are classified into 3,135 categories (WordNet synsets). In this report we describe the ShapeNet effort as a whole, provide details for all currently available datasets, and summarize future plans.
</details>

### Towards Understanding Articulated Objects [RSSW 2009, Sturm etc.]
[📄 Paper](http://ais.informatik.uni-freiburg.de/publications/papers/sturm09rss-manip.pdf)
- Task: Articulation
- Stages: Model Tamplates
- Input: Annitated RGB image
<details open>
<summary><b>Abstract</b></summary>
<br>

Robots operating in home environments must be able to interact with articulated objects such as doors or drawers. Ideally, robots are able to autonomously infer articulation models by observation. In this paper, we present an approach to learn kinematic models by inferring the connectivity of rigid parts and the articulation models for the corresponding links. Our method uses a mixture of parameterized and parameter-free representations. To obtain parameter-free models, we seek for low-dimensional manifolds of latent action variables in order to provide the best explanation of the given observations. The mapping from the constrained manifold of an articulated link to the work space is learned by means of Gaussian process regression. Our approach has been implemented and evaluated using real data obtained in various home environment settings. Finally, we discuss the limitations and possible extensions of the proposed method.
</details>

