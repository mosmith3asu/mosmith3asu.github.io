---
title: Research Question
---
# Bio-inspired Passive Power Attenuation Mechanism for Jumping Robot
[Return Home](/index)

# Research Question

## Introduction
What is your research question?

How can a jumping biped robot be constructed to dissipate kinetic energy upon landing and passively achieve a stable standing position?

## 1. Approach (Tractability)

The research question that is being asked was specifically chosen so that it may avoid requirements that do not directly pertain to or significantly contribute to the topic of interest.  There are many different methods for dissipating kinetic energy and even more contexts in which doing so would be useful but the 15-week course implies that limiting the scope of this project to only focus on passive mechanisms operating in simplified conditions.  Several constraints were defined to address this and make this project more feasible to complete with the limited amount of resources available.

The first set of constraints pertains to the bipedal robot that will be used as a testing platform.  This robot will be a rigid body in the form of a bipedal robot that is between 4 and 6 inches tall. It will not include actuators at any of the joints.  This is done under the assumption that the impact of a functional bipedal robot can be adequately simulated without the actuated joints.  This simplification removes the need for designing and implementing a fully actuated system and allows this project to concentrate on the topic of research.  This robotic platform will be a constant mass through all of the experiments to limit the dimensionality of the investigated variables.  The selected mass value will be under 1 lb in order to represent a small robot.   Also, the robot will be laterally constrained under the assumption that the legs, that are laterally located from each other, in a functional robot would be able to provide a means for balancing around this axis and is therefore not a significant consideration for valid testing of this system.

The second set of constraints involves manufacturing and testing methods.  Both the robotic platform and the testing equipment will be made using rapid prototyping methods like 3-D printing, cardboard mockups and laser cut assemblies.  This decision was made to address the limited budget and manufacturing issues that could arise. The passive mechanism that dissipates the kinetic energy will be made using a laminate construction process to address these same concerns.  During testing, the robotic platform will be dropped from a known height and impact the landing platform so that the vertical axis of the robot is normal to the platform.  This constraint is specified to reduce the complexity of collecting all the variations of impact orientations for each of our tested devices.  However, if time allows, this constraint may be relaxed to include non-ideal impact orientations.

Dual-stiffness materials represent a promising method for preventing failure under high impact loads, and will serve as a starting point for the design of the test platform. An example of a dual-stiffness material found in nature is with wasp wings. The structure of the wings contains cuticles, which are rigid, and resilin, which acts like a spring and connects the cuticles. Upon impact, the resilin in the wings fold while the cuticles remain rigid to prevent buckling and then spring back into a rigid state. [1] This approach of bio-inspiration can be applied to our project to create dual-stiffness joints that are able to comply and while landing but remain still for normal locomotion.
Another dual-stiffness system found in nature is the human leg. In order to adapt to changing terrain, the ankle and knee modulation affects the stiffness. The muscles within the leg are able to dynamically create equilibrium or nonequilibrium states depending on the activity. [2] A passive system that absorbs shock and helps with stability is the ostrich’s foot. It has viscoelastic layers that allow shock to be absorbed within the tissue without high deformation. For robotics, this can be simplified to a multi-layered foot with each layer having a material with different levels of elastic moduli. [3] Using this bio-inspiration would be a simple way to absorb shock within the constraints of this course.

## 2. Background (Novelty)
To establish the novelty of this project, a literature review was conducted by searching for different combinations of classes of robotic systems and research goals. The primary keywords used are listed in Table 1.
#### Table 1: Literature review keywords
|Classes of System|Classes of Research Goals|
|-|-|
|laminate robot|drop test|
|biped robot|landing control|
|one legged robot|shock absorption|
|jumping robot|impact absorption|
|hopping robot|impact attenuation|
|compliant legs||

Previous work has explored the use of jumping to allow small robots to traverse large obstacles [4]. This work was motivated by a need for small robots that can be used in extreme environments for Earth and space exploration. Jumping is presented as a locomotion strategy that can be added to wheeled robots and that is more robust than a complete walking robot. In the prototype systems, the impact forces from landing are addressed by the use of robust materials and structure. The landing orientation of the system is addressed by the addition of actuators that allow the robot to self-right. While this system is effective, it cannot maintain a desired orientation during landing, and it requires metal frame components that make it relatively heavy for its size (3 lbs, 6 x 6 x 6 in3) since the entire structure must absorb the landing impact.

For robots that perform repetitive jumping motions, the addition of a spring loaded joint can contribute to absorbing the impact forces and allow the energy from landing to be stored and released during the next jump. The work presented in [5] demonstrates a hopping robot that uses this design approach and a software controller to absorb, store, and release energy at the proper points in the gait pattern with a minimal number of actuators. Adding a passive compliant joint in the manner presented reduces the forces that the leg links must absorb, although high torques are still present at revolute joints. Since there is no damping present in the system, however, it is only suitable for repetitive motions where stored energy can be released as part of the motion cycle. The presented system represents a large robot (30 lbs, 20 in tall), but the concepts also can be applied to small robots [6].

Jumping can also be approached by using actuated joints to both control landing forces and maintain system balance [7]. This system uses a padded foot to absorb some of the impact, but most of the forces must be dissipated using the actuators. The presented work focuses on the stability analysis used to determine joint torques required for take off and landing. The need to have an actuator at each joint of the leg increases the size and complexity of the system compared to underactuated legs with a spring loaded joint.

The benefits of spring-loaded and controlled joints can be combined  by the addition of an actuator with an elastic power transmission component [8]. This configuration allows the elastic joint to passively absorb the initial impact with a faster response time than a motor can provide. Shortly after impact, the actuator can change the torque provided by the spring loaded joint to further reduce forces on the robot and prevent the system from bouncing.

Prior work has previously explored achieving stable landings with jumping robots. However, this work often uses bulky mechanical assemblies and actuated leg joints. This can make these methods difficult to apply to small robotic systems. Our work will investigate how laminate construction techniques can be used to create a system that can achieve a stable landing using lightweight materials and no actuation. It is intended for robots in the range of 4-6 in tall and less than 1 lb, making full actuation and complex mechanisms impractical. It is also intended to apply to robots which must perform a single jump, so the developed system cannot store and release energy with a simple spring loaded joint as in previous work with small hopping robots.

## 3. Impact (Interesting)
As discussed in the previous section, jumping can provide a way for small robots to navigate over large obstacles that they would not otherwise be able to climb. Jumping can also be combined with a gliding system to quickly travel large horizontal distances [9]. However, landing after a jump is not adequately addressed in small robots. Robots in current literature rely on either a strong frame that can withstand a large impact, or on actuated joints which increase size and introduce a complex controls problem. In addition, there is not a strong surrounding body of literature regarding solving the problem of landing using laminate fabrication techniques, which are of interest for creating systems that are low cost and can be rapidly manufactured. Solutions to our research question would provide a simple and efficient means for stabilization of high impact forces in small robots. This would support the development of jumping as a feasible mode of locomotion for low cost robots.

In the last 10 years, laminate construction methods have been applied to building robots of size around 1 to 10 centimeters such as [10], [11], and [12]. These studies demonstrate that such a method is accessible, scalable, and can generate high performance but low-cost robots. At the same time, construction, modelling, and control of bipedal robots are also being developed and even commercialized such as [13], [14], and [15]. Combining the knowledge and experience gained from both fields, it is now possible and reasonable to explore the functions and properties of lamination structures or mechanisms applied to bipedal robots. Although only the stable landing is researched due to the constraint of a class project, modeling method, effect of inherited compliance of laminates, and other insights gained could point direction for the development of a foldable bipedal robot in the future.

Our results would be most useful to anyone designing a jumping bipedal robot as that is what our results will be focused on. However, the results will also be useful to any robot that needs to reduce stress upon coming in contact with a surface. This could be for other pedal robots or for other small robots that need to land in an emergency.
Considering the cost constraints of the course, the final mechanism will be relatively low in cost. In applications where low cost is important, like in educational products, swarm robotics, and disposable robotics, the results of our work could be useful for enabling the design of effective jumping systems.
Due to the nature of our approach heavily relying on bio-inspiration of dual-stiffness materials, there is potential for the idea to come full circle and be used in the field of prosthetics. The benefits of flexible joints doing negative work upon impact while still staying rigid could make prosthetics more similar to the functionality of their biological counterparts.

By advancing the field of pedal robots, their adoption could become more widespread. Their exact use and impact on the wider society is unknown at this time, but there is potential for them to make a big impact as they are applied in more applications.

## 4. Future Work (Open-Ended)
A majority of the constraints that were defined in this project were defined so that this project is reasonable to complete with the given resources; mainly time.  Therefore, this project has many additional areas of investigation if time allows.  Possibly one of the strongest constraints is that the robot will impact the landing platform in a single, ideal orientation.  This allows, if the time is available, the possibility of testing the response of the proposed mechanisms at different impact angles. The scope of testing can be further expanded by varying the mass of the robotic platform to observe the response of the mechanisms under different magnitudes of stress.  Also, different manufacturing methods and materials can be investigated to observe their effects on the mechanisms that are designed during the initial investigation. 
Further investigation can also be performed as an expansion of this research.  Mainly, our robotic platform is a rigid body and can be improved by incorporating the response of actuated joints.  This is almost definitely out of the scope of this class but is a relevant area of research that would contribute to the proposed research question.

## 5. Applications (Modular)
While the focus of our question centers around the landing stability of a jumping biped robot, the solution, if found to be successful, can also be applied to existing robots with legs. Whether their functionality is primarily focused on walking, jumping, or something else, creating a more stable landing when a leg comes in contact with the ground is beneficial. Additionally, the methods explored in the question can be expanded to robots that are not bipedal. This method of compliance with laminated materials has already been applied to a robotic gripper and drone, so it could be applied to more robotic applications that are not pedal. 
The application does not have to be restricted just to the field of robotics. Any application that requires a mechanism to stay stiff while complying under high-impact conditions could be adapted accordingly. Perhaps in the field of automobiles, making more compliant parts that are able to restore to an original state rather could reduce waste while not compromising on safety.

## 6. Team Fit
Answering this research question requires study and gaining inspiration from the bipedal locomotion in nature. Biological structures and mechanisms are still quite different from existing robot designs. Our team is interested in learning from nature to propose novel bioinspired design ideas. Moreover, kinematic, dynamic, and material modelling are necessary to understand the theory behind a stable landing as well as implementing it into an actual robotic design. With various previous experiences in such areas and the new tools and methods learned from this course, our team can develop a reasonably accurate mathematical model. Finally, we have sufficient rapid prototyping knowledge and tools such as 3D printing, cardboard crafting, and basic electronics building. Our team not only will be able to develop the proof-of-concept prototype and testing rig, but also can perform experiments and collect quantitative data for analysis.

## 7. Class Fit
Passive stable landing is more important for low-cost bipedal robots because it can greatly reduce the mechanical and electrical complexity of the system while providing acceptable performance. Using foldable robotics techniques to solve this question fits the general goal of such kinds of robots and can also provide insights on solving other bipedal locomotion issues. Furthermore, robots constructed using such techniques are already more compliant than those built with traditional techniques mainly due to the materials. Instead of trying to get rid of the softness, it may be more efficient to use it to store and dissipate energy to achieve mechanical intelligence for some tasks. Additionally, foldable techniques provide a fast and accessible way of building functional robots, which eases the tools and time limitations of this course project. 

## References
[1]
Mintchev, S., Shintake, J., & Floreano, D. (2018). Bioinspired dual-stiffness origami. Science Robotics, 3(20), eaau0275. https://doi.org/10.1126/scirobotics.aau0275

[2]
Kim, W., João, F., Tan, J., Mota, P., Vleck, V., Aguiar, L., & Veloso, A. (2013). The natural shock absorption of the leg spring. Journal of Biomechanics, 46(1), 129–136. https://doi.org/10.1016/j.jbiomech.2012.10.041

[3]
Han, D., Zhang, R., Yu, G., Jiang, L., Li, D., & Li, J. (2020). Study on bio-inspired feet based on the cushioning and shock absorption characteristics of the ostrich foot. PLOS ONE, 15(7), e0236324. https://doi.org/10.1371/journal.pone.0236324

[4]
Fiorini, P., & Burdick, J. (2003). The development of hopping capabilities for small robots. Autonomous Robots, 14(2–3), 239–254. https://doi.org/10.1023/A:1022239904879

[5]
Hyon, S. H., Emura, T., & Mita, T. (2003). Dynamics-based control of a one-legged hopping robot. Proceedings of the I MECH E Part I Journal of Systems & Control Engineering, 217(2), 83–98. https://doi.org/10.1243/095965103321512800

[6]
Pfeifer, R., & Gómez, G. (2009). Morphological Computation – Connecting Brain, Body, and Environment. In Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics) (Vol. 5436, Issue January 2006, pp. 66–83). https://doi.org/10.1007/978-3-642-00616-6_5

[7]
Goswami, D., & Vadakkepat, P. (2009). Planar Bipedal Jumping Gaits With Stable Landing. IEEE Transactions on Robotics, 25(5), 1030–1046. https://doi.org/10.1109/TRO.2009.2026502

[8]
Dallali, H., Kormushev, P., Tsagarakis, N. G., & Caldwell, D. G. (2014). Can active impedance protect robots from landing impact? 2014 IEEE-RAS International Conference on Humanoid Robots, 2015-Febru, 1022–1027. https://doi.org/10.1109/HUMANOIDS.2014.7041490

[9]
Gadekar, V. P. (2020). Design of Wings for Jump Gliding in a Biped Robot. Thesis. Arizona State University.

[10]
A. M. Hoover, S. Burden, S. Shankar Sastry, and R. S. Fearing, “Bio-inspired design and dynamic maneuverability of a minimally actuated six-legged robot,” in 2010 3rd IEEE RAS & EMBS international conference on biomedical robotics and biomechatronics, 2010, pp. 869–876.

[11]
D. W. Haldane, K. C. Peterson, F. L. Garcia Bermudez, and R. S. Fearing, “Animal-inspired design and aerodynamic stabilization of a hexapedal millirobot,” in 2013 IEEE international conference on robotics and automation, 2013, pp. 3279–3286.

[12]
A. T. Baisch, O. Ozcan, B. Goldberg, D. Ithier, and R. J. Wood, “High speed locomotion for a quadrupedal microrobot,” The International Journal of Robotics Research, May 2014.

[13]
Z. Xie, G. Berseth, P. Clary, J. Hurst and M. van de Panne, "Feedback Control For Cassie With Deep Reinforcement Learning," 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Madrid, 2018, pp. 1241-1246, doi: 10.1109/IROS.2018.8593722.

[14]
S. Feng, E. Whitman, X. Xinjilefu and C. G. Atkeson, "Optimization based full body control for the atlas robot," 2014 IEEE-RAS International Conference on Humanoid Robots, Madrid, 2014, pp. 120-127, doi: 10.1109/HUMANOIDS.2014.7041347.

[15]
Radford, N.A., Strawser, P., Hambuchen, K., Mehling, J.S., Verdeyen, W.K., Donnan, A.S., Holley, J., Sanchez, J., Nguyen, V., Bridgwater, L., Berka, R., Ambrose, R., Myles Markee, M., Fraser‐Chanpong, N.J., McQuin, C., Yamokoski, J.D., Hart, S., Guo, R., Parsons, A., Wightman, B., Dinh, P., Ames, B., Blakely, C., Edmondson, C., Sommers, B., Rea, R., Tobler, C., Bibby, H., Howard, B., Niu, L., Lee, A., Conover, M., Truong, L., Reed, R., Chesney, D., Platt, R., Jr, Johnson, G., Fok, C.‐L., Paine, N., Sentis, L., Cousineau, E., Sinnet, R., Lack, J., Powell, M., Morris, B., Ames, A. and Akinyode, J. (2015), Valkyrie: NASA's First Bipedal Humanoid Robot. J. Field Robotics, 32: 397-419. https://doi.org/10.1002/rob.21560
