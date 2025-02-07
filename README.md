Step 1. capture scene flow F/F*.

dir:fss-gr-sceneflow

(1)Environment configuration
V100-32G,cuda=10.1,python=3.6.13

install_environment.sh:
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install tensorboard
pip install tqdm==4.60.0
#installmayaviforvisualization
pip install vtk==9.1.0
pip install pyQt5==5.15.2
pip install mayavi==4.7.4
sudo apt-get install rar
pip install thop

(2)Compile the Chamfer Distance op
sh compile_chamfer_distance_op.sh

(3)./data/FlowNet3D/SHREC2017/test_gnpz2.py
construct pairs of point clouds.
Run the python file above to generate 89600 npz files.

(4)scripts/train_on_shrec.sh
The above three parameters can be modified.
Recon_flow and its data source are generated on the upper side: stored in /recon_flow/log.txt and /data_source_recon_flow/log.txt respectively. The generation model is saved in log_dir.
recon_flow is F.

(5)scripts/evaluate_on_shrec.sh
Run evaluate_on_shrec.sh to generate est_flow and its data source.

(6)scripts/process_est_flow.py
Running `scripts/process_est_flow.py` generates 2800 txt files under \est_flow.
est_flow is F*.

Finally, scene flow is obtained.

Step 2.Fusing features.

dir:fss-gr-s-d

(1)Environment configuration

v100-16G,Cuda10.0,python=3.6
conda activate
conda create -n name python=3.6
conda activate name

conda install numpy==1.16
pip install   matplotlib
pip install pyyaml
conda install pytorch-cpu==1.1.0 torchvision-cpu==0.3.0 cpuonly -c pytorch
pip install -i https://pypi.doubanio.com/simple/ tensorflow-gpu==1.13.1

(2)Dataset transmission
Put the `est_flow` or 'recon_flow' (F* or F)generated in Step 1 into the `dataset` folder of fss-gr-s-d.

(3)Compile Customized TF Operators


(4)train/test.

cd shrec2017_release

bash train_dynamic.sh

bash test.sh


Our code mainly refers to the following two papers. We are very grateful for that.

Zhong, J.X.; Zhou, K.; Hu, Q.;Wang, B.; Trigoni, N.; Markham, A. No Pain, Big Gain: Classify Dynamic Point Cloud Sequences 598
with Static Models by Fitting Feature-level Space-time Surfaces. In Proceedings of the 2022 IEEE/CVF Conference on Computer 599
Vision and Pattern Recognition (CVPR), 2022, pp. 8500–8510. https://doi.org/10.1109/CVPR52688.2022.00832.

Lang, I.; Aiger, D.; Cole, F.; Avidan, S.; Rubinstein, M. SCOOP: Self-Supervised Correspondence and Optimization-Based Scene
Flow. In Proceedings of the 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Vancouver, BC,
Canada, 17–24 June 2023; pp. 5281–5290. https://doi.org/10.1109/CVPR52729.2023.00511.



