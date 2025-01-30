1.fss-gr-sceneflow
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
