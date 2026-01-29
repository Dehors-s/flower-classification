import os
import zipfile
from aistudio_sdk.hub import upload_folder

os.environ["AISTUDIO_ACCESS_TOKEN"] = "3655a4e925b5267ff6bf524117c6578fe5ff281b"


def split_and_upload(local_path, repo_id, target_path):
    """如果文件夹很大，可以先压缩再上传"""

    # 创建临时zip文件
    zip_filename = "flower_data_temp.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(local_path):
            for file in files:
                file_path = os.path.join(root, file)
                # 在zip中保持相对路径
                arcname = os.path.relpath(file_path, local_path)
                zipf.write(file_path, arcname)

    print(f"压缩完成，文件大小: {os.path.getsize(zip_filename) / 1024 / 1024:.2f} MB")

    try:
        # 上传压缩包
        res = upload_folder(
            repo_id=repo_id,
            folder_path='.',  # 当前目录
            path_in_repo=target_path,
            commit_message='上传压缩的花卉数据集',
            repo_type='dataset'
        )
        print("上传成功！")
        return res
    except Exception as e:
        print(f"上传失败: {e}")
        return None
    finally:
        # 清理临时文件
        if os.path.exists(zip_filename):
            os.remove(zip_filename)


# 使用分块上传
split_and_upload(
    r'D:\ptcharm\project\花卉分析\train',
    'Dehors/DeiTIIILarge2',
    'flower_train_data/'
)