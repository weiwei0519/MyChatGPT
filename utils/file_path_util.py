# coding=UTF-8
# 关于path和file的工具类

'''
@File: file_path_util
@Author: WeiWei
@Time: 2023/3/11
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import os
import shutil
import glob


def tgt_comp_del(file, tgt_path, replace, pattern='file'):
    for fpath, fdir, ffile in os.walk(tgt_path):
        if fpath == tgt_path:  # 只处理目标路径下的子路径和文件，其它都不对比
            if pattern == 'file' and len(fdir) == 0 and file in ffile:
                # 有重名文件
                if replace:
                    os.remove(os.path.join(fpath, file).replace('\\', '/'))
                else:
                    name, ext = file.split('.')
                    new_file = name + '_old.' + ext
                    os.renames(os.path.join(fpath, file).replace('\\', '/'),
                               os.path.join(fpath, new_file).replace('\\', '/'))
            if pattern == 'dir' and len(ffile) == 0 and file == fdir[0]:
                # 有重名目录
                if replace:
                    shutil.rmtree(os.path.join(fpath, file).replace('\\', '/'))
                else:
                    new_file = file + '_old'
                    os.renames(os.path.join(fpath, file).replace('\\', '/'),
                               os.path.join(fpath, new_file).replace('\\', '/'))


# 移动src_path目录下所有的文件，到tgt_path目录下，若tgt_path下有同名文件，则覆盖
def move_files(src_path, tgt_path, replace=True):
    for dirpath, dirnames, filenames in os.walk(src_path):
        dirpath = eval(repr(dirpath).replace('\\', '\\\\'))
        # 情况1：当前walk到文件list，并且是src_path目录下的文件，则遍历filenames并move到tgt_path下
        if src_path == dirpath.replace('\\', '/') and len(dirnames) == 0 and len(filenames) > 0:
            for file in filenames:
                tgt_comp_del(file, tgt_path, replace, pattern='file')
                shutil.copy2(os.path.join(dirpath, file).replace('\\', '/'), tgt_path)
        # 情况2：当前walk到文件夹，并且是src_path目录下的文件夹，则将此文件夹整体move到tgt_path下
        if len(dirnames) > 0 and len(filenames) == 0:
            for dir in dirnames:
                tgt_comp_del(dir, tgt_path, replace, pattern='dir')
                shutil.copy2(os.path.join(dirpath, dir).replace('\\', '/'), tgt_path)


if __name__ == '__main__':
    model_dir = "../models/chatgpt-aia-chinese/gpt-aia-chinese"
    # 训练结束后，将最后一个checkpoint的整体模型参数文件，复制到model dir output目录
    checkpoint = glob.glob(os.path.join(model_dir, 'checkpoint-*'))
    if len(checkpoint) > 0:
        checkpoint = (checkpoint[0]).replace("\\", "/")
        move_files(checkpoint, model_dir)
