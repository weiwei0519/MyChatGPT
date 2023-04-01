# 启动服务
conda activate base
cd /data/AI_project/MyChatGPT
streamlit run answer_rank_labeler.py --server.port 8904

# 重启服务
ps -aux | grep python
kill -9 python_pid
cd /data/AI_project/MyChatGPT
streamlit run answer_rank_labeler.py --server.port 8904


streamlit run chatglm_web.py --server.port 8905

/home/inoladm01/python/anaconda/enter/bin/python  /data/AI_project/MyChatGPT/answer_rank_labeler.py

export PATH=$PATH:/home/inoladm01/python/anaconda/enter/bin
conda-env list
conda activate base  # 激活anaconda环境。

watch -n 10 nvidia-smi

# Linux安装gcc
# 按顺序单个安装
rpm -ivh gmp-4.1.4-12.3_2.el5.x86_64.rpm --force --nodeps
rpm -ivh ppl-0.10.2-11.el6.x86_64.rpm --force --nodeps
rpm -ivh cloog-ppl-0.15.7-1.2.el6.x86_64.rpm --force --nodeps
rpm -ivh mpfr-2.4.1-6.el6.x86_64.rpm --force --nodeps
rpm -ivh cpp-4.4.7-4.el6.x86_64.rpm --force --nodeps
rpm -ivh kernel-headers-2.6.32-431.el6.x86_64.rpm --force --nodeps
rpm -ivh glibc-headers-2.12-1.132.el6.x86_64.rpm --force --nodeps
rpm -ivh glibc-devel-2.12-1.132.el6.x86_64.rpm --force --nodeps
rpm -ivh gcc-4.4.7-4.el6.x86_64.rpm --force --nodeps
rpm -ivh libstdc++-devel-4.4.7-4.el6.x86_64.rpm --force --nodeps
rpm -ivh gcc-c++-4.4.7-4.el6.x86_64.rpm --force --nodeps
# 或直接批量安装
rpm  -ivh  *.rpm --nodeps --force