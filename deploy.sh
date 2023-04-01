#生产的requirements文件中，需要删除掉distribute，pip，setuptools，wheel这几项package。
pip list --format=freeze > requirements.txt

# 作用范围：当前项目使用的类库导出生成为requirements.txt。
# 使用方法：pipreqs 加上当前路径即可。在导出当前项目使用的类库时，先定位到项目根目录，然后调用 pipreqs ./ --encoding=utf8 命令，该命令避免编码错误，并自动在根目录生成 requirements.txt 文件。
pip install pipreqs
pipreqs --use-local --encoding=utf8 --force

#添加$PATH
export PATH=$PATH:/home/inoladm01/python/anaconda/enter/bin
echo $PATH