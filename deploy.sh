#生产的requirements文件中，需要删除掉distribute，pip，setuptools，wheel这几项package。
pip list --format=freeze > requirements.txt