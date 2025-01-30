![image](https://github.com/user-attachments/assets/36936f8a-0e70-4594-a146-dfcb407feae9)1.fss-gr-sceneflow
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






2./data/FlowNet3D/SHREC2017/test_gnpz2.py
Run the python file above to generate 89600 npz files.
