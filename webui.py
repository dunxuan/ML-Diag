# import os
import gradio as gr
import pandas as pd

# 当前程序所在目录
# PATH_DIRNAME = os.path.dirname(__file__)

# 在上传数据集时使用该全局变量存储训练集
df_train = pd.DataFrame()


# 处理上传训练集的函数
def upload_csv(file):
    global df_train
    df_train = pd.read_csv(file)
    head = df_train.head()
    return head


with gr.Blocks() as demo:
    with gr.Tab('训练'):
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    # 选择模型
                    # 上传模型
                    # 上传csv文件
                    upload_csv_button = gr.UploadButton(label="上传csv文件",
                                                        variant='primary',
                                                        # file_types=['.csv']
                                                        )
            with gr.Column():
                with gr.Box():
                    csv_display = gr.Textbox(placeholder='请上传csv文件并在此处检验',
                                             label='检验数据集',
                                             interactive=False,
                                             show_copy_button=True)

        upload_csv_button.upload(fn=upload_csv, inputs=upload_csv_button, outputs=csv_display)

if __name__ == '__main__':
    demo.queue()
    demo.launch()
