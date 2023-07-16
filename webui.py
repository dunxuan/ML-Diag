import json
import os
from datetime import datetime

import gradio as gr
import pandas as pd
from skops.io import dump

from classifier import Classifier

# 当前程序所在目录
# PATH_DIRNAME = os.path.dirname(__file__)

# 选择的预训练模型
selected_model = None
# 模型菜单
model_upload = ['不加载模型']
model_download = []
# json
json_download = []


# 初始化模型菜单model_list
def model_list_init():
    # 初始化模型菜单
    path = './models/'
    files = os.listdir(path)
    for file in files:
        if file.endswith('.skops'):
            model_upload.append(os.path.join(path, file))


# 控制clf_params_input变量的输入
def clf_params_non_interactive(interactive):
    interactive = not interactive
    return [clf_params_input['n_estimators'].update(interactive=interactive),
            clf_params_input['criterion'].update(interactive=interactive),
            clf_params_input['max_depth'].update(interactive=interactive),
            clf_params_input['min_samples_split'].update(interactive=interactive),
            clf_params_input['min_samples_leaf'].update(interactive=interactive),
            clf_params_input['min_weight_fraction_leaf'].update(interactive=interactive),
            clf_params_input['max_features'].update(interactive=interactive),
            clf_params_input['max_leaf_nodes'].update(interactive=interactive),
            clf_params_input['min_impurity_decrease'].update(interactive=interactive),
            clf_params_input['bootstrap'].update(interactive=interactive),
            clf_params_input['oob_score'].update(interactive=interactive),
            clf_params_input['n_jobs'].update(interactive=interactive),
            clf_params_input['random_state'].update(interactive=interactive),
            clf_params_input['verbose'].update(interactive=interactive),
            clf_params_input['warm_start'].update(interactive=interactive),
            clf_params_input['class_weight'].update(interactive=interactive),
            clf_params_input['ccp_alpha'].update(interactive=interactive),
            clf_params_input['max_samples'].update(interactive=interactive)]


# 处理'选择模型'菜单的选项改变
def select_model(name):
    global selected_model
    if name is None or name == '不加载模型':
        selected_model = None
        return [use_model_params.update(value=False, interactive=False)] + clf_params_non_interactive(False)
    else:
        selected_model = name
        return [use_model_params.update(value=True, interactive=True)] + clf_params_non_interactive(True)


# 处理'上传模型(*.skops)'
def model_add(uploaded_model_file):
    global model_upload
    model_upload.append(uploaded_model_file.name)
    return choice_name.update(choices=model_upload)


# 处理上传数据集
def upload_file(uploaded_file):
    train_file = uploaded_file.name
    head = pd.read_csv(train_file).head()
    return head


# 处理'数据集分割方法'
def cv_method_change(method):
    visible = method == '交叉验证法'
    return [cv_params_input['n_splits'].update(visible=visible),
            cv_params_input['shuffle'].update(visible=visible),
            cv_params_input['random_state'].update(visible=visible),
            cv_params_input['p'].update(visible=not visible)]


# 训练
def training(uploaded_train_file, cv_method, cvp0, cvp1, cvp2, cvp3, use_model_params, clfp0, clfp1, clfp2,
             clfp3, clfp4, clfp5, clfp6, clfp7, clfp8, clfp9, clfp10, clfp11, clfp12, clfp13, clfp14, clfp15, clfp16,
             clfp17):
    if uploaded_train_file is None:
        raise gr.Error('请先上传训练集')
    cv_params = {'n_splits': int(cvp0),
                 'shuffle': cvp1,
                 'random_state': int(cvp2) if cvp2 else None,
                 'p': int(cvp3)}
    clf_params = {'n_estimators': int(clfp0),
                  'criterion': clfp1,
                  'max_depth': int(clfp2) if clfp2 else None,
                  'min_samples_split': float(clfp3) if '.' in clfp3 else int(clfp3),
                  'min_samples_leaf': float(clfp4) if '.' in clfp4 else int(clfp4),
                  'min_weight_fraction_leaf': clfp5,
                  'max_features': clfp6,
                  'max_leaf_nodes': clfp7 if clfp7 else None,
                  'min_impurity_decrease': clfp8,
                  'bootstrap': clfp9,
                  'oob_score': clfp10,
                  'n_jobs': int(clfp11),
                  'random_state': int(clfp12) if clfp12 else None,
                  'verbose': int(clfp13),
                  'warm_start': clfp14,
                  'class_weight': clfp15 if clfp15 else None,
                  'ccp_alpha': clfp16,
                  'max_samples': (float(clfp17) if '.' in clfp17 else int(clfp17)) if clfp17 else None}
    global selected_model
    train_file = uploaded_train_file.name
    clf = Classifier(df_file=train_file, model_file=selected_model)
    scores, model = clf.train(cv_method=cv_method,
                              cv_params=cv_params,
                              use_model_params=use_model_params,
                              clf_params=clf_params)
    # 处理scores
    metrics_mean = {'F1_macro平均值': scores.mean(),
                    '标准差': scores.std()}
    metrics = {}
    for i in range(len(scores)):
        metrics[f'子集 {i}'] = scores[i]
    # 处理model_file
    now = datetime.now()
    model_name = f"./Cache/model_{now.strftime('%y%m%d_%H%M%S')}.skops"
    dump(model, model_name)
    global model_download
    model_download.append(model_name)
    model_file_update = model_file.update(value=model_download)
    return [metrics_mean, metrics, model_file_update]


# 处理验证
def validate(uploaded_validate_file):
    global selected_model
    if selected_model is None:
        raise gr.Error('请选择模型')
    validate_file = uploaded_validate_file.name
    clf = Classifier(df_file=validate_file, model_file=selected_model)
    y_pred, validate_scores = clf.validate()
    metrics = {}
    metrics['准确度'] = validate_scores
    json_dict = {}
    for i in range(len(y_pred)):
        json_dict[str(i)] = int(y_pred[i])
    now = datetime.now()
    json_name = f"./Cache/validate_{now.strftime('%y%m%d_%H%M%S')}.json"
    with open(json_name, 'w') as f:
        json.dump(json_dict, f)
    global json_download
    json_download.append(json_name)
    json_file_update = json_file.update(value=json_download)
    json_display_update = json_display.update(value=json_dict)
    return [metrics, json_file_update, json_display_update]


# 处理测试
def test(uploaded_test_file):
    global selected_model
    if selected_model is None:
        raise gr.Error('请选择模型')
    test_file = uploaded_test_file.name
    clf = Classifier(df_file=test_file, model_file=selected_model)
    y_pred = clf.test()
    json_dict = {}
    for i in range(len(y_pred)):
        json_dict[str(i)] = int(y_pred[i])
    now = datetime.now()
    json_name = f"./Cache/test_{now.strftime('%y%m%d_%H%M%S')}.json"
    with open(json_name, 'w') as f:
        json.dump(json_dict, f)
    global json_download
    json_download.append(json_name)
    json_file_update = json_file.update(value=json_download)
    json_display_update = json_display.update(value=json_dict)
    return [json_file_update, json_display_update]


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Box():
            with gr.Row():
                model_list_init()
                # 选择模型
                choice_name = gr.Dropdown(choices=model_upload,
                                          value='不加载模型',
                                          multiselect=False,
                                          label='选择模型',
                                          scale=5,
                                          allow_custom_value=True)
                # 上传模型(*.skops)
                upload_model_btn = gr.UploadButton(label='上传模型(*.skops)',
                                                   scale=1,
                                                   file_types=['.skops'])

    with gr.Tab('训练'):
        with gr.Row():
            # 上传训练集(*.csv)
            upload_train_btn = gr.UploadButton(label='上传训练集(*.csv)',
                                               variant='primary',
                                               file_types=['.csv'])
            train_btn = gr.Button(value='训练',
                                  variant='primary')
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Accordion(label='分割数据集'):
                        # 数据集分割方法
                        cv_method_input = gr.Radio(choices=['交叉验证法', '留一法'],
                                                   value='交叉验证法',
                                                   label='分割方法')
                        # 数据集分割参数
                        cv_params_input = {'n_splits': 5, 'shuffle': False, 'random_state': None, 'p': 1}
                        cv_params_input['n_splits'] = gr.Number(value=cv_params_input['n_splits'], label='n_splits',
                                                                minimum=2)
                        cv_params_input['shuffle'] = gr.Checkbox(value=cv_params_input['shuffle'], label='shuffle')
                        cv_params_input['random_state'] = gr.Textbox(value=cv_params_input['random_state'],
                                                                     label='random_state')
                        cv_params_input['p'] = gr.Number(value=cv_params_input['p'], label='p', visible=False)
                with gr.Row():
                    with gr.Accordion(label='训练参数'):
                        with gr.Row():
                            # 分类器参数
                            clf_params_input = {'n_estimators': 100,
                                                'criterion': 'gini',
                                                'max_depth': None,
                                                'min_samples_split': 2,
                                                'min_samples_leaf': 1,
                                                'min_weight_fraction_leaf': 0.0,
                                                'max_features': 'sqrt',
                                                'max_leaf_nodes': None,
                                                'min_impurity_decrease': 0.0,
                                                'bootstrap': True,
                                                'oob_score': False,
                                                'n_jobs': -1,
                                                'random_state': None,
                                                'verbose': 0,
                                                'warm_start': False,
                                                'class_weight': None,
                                                'ccp_alpha': 0.0,
                                                'max_samples': None}
                            use_model_params = gr.Checkbox(
                                value=False,
                                label='使用模型参数',
                                interactive=False)
                        with gr.Row():
                            clf_params_input['n_estimators'] = gr.Number(
                                value=clf_params_input['n_estimators'],
                                label='n_estimators',
                                interactive=True)
                            clf_params_input['criterion'] = gr.Dropdown(
                                choices=['gini', 'entropy', 'log_loss'],
                                value=clf_params_input['criterion'],
                                label='criterion',
                                interactive=True)
                            clf_params_input['max_depth'] = gr.Textbox(
                                value=clf_params_input['max_depth'],
                                label='max_depth',
                                interactive=True)
                            clf_params_input['min_samples_split'] = gr.Textbox(
                                value=clf_params_input['min_samples_split'],
                                label='min_samples_split',
                                interactive=True)
                            clf_params_input['min_samples_leaf'] = gr.Textbox(
                                value=clf_params_input['min_samples_leaf'],
                                label='min_samples_leaf',
                                interactive=True)
                            clf_params_input['min_weight_fraction_leaf'] = gr.Number(
                                value=clf_params_input['min_weight_fraction_leaf'],
                                label='min_weight_fraction_leaf',
                                interactive=True)
                            clf_params_input['max_features'] = gr.Dropdown(
                                choices=['sqrt', 'log2', None],
                                value=clf_params_input['max_features'],
                                label='max_features',
                                interactive=True)
                            clf_params_input['max_leaf_nodes'] = gr.Textbox(
                                value=clf_params_input['max_leaf_nodes'],
                                label='max_leaf_nodes',
                                interactive=True)
                            clf_params_input['min_impurity_decrease'] = gr.Number(
                                value=clf_params_input['min_impurity_decrease'],
                                label='min_impurity_decrease',
                                interactive=True)
                            clf_params_input['n_jobs'] = gr.Number(
                                value=clf_params_input['n_jobs'],
                                label='n_jobs',
                                interactive=True)
                            clf_params_input['random_state'] = gr.Textbox(
                                value=clf_params_input['random_state'],
                                label='random_state',
                                interactive=True)
                            clf_params_input['verbose'] = gr.Number(
                                value=clf_params_input['verbose'],
                                label='verbose',
                                interactive=True)
                            clf_params_input['class_weight'] = gr.Dropdown(
                                choices=['balanced', 'balanced_subsample', None],
                                value=clf_params_input['class_weight'],
                                label='class_weight',
                                interactive=True)
                            clf_params_input['ccp_alpha'] = gr.Number(
                                value=clf_params_input['ccp_alpha'],
                                label='ccp_alpha',
                                interactive=True)
                            clf_params_input['max_samples'] = gr.Textbox(
                                value=clf_params_input['max_samples'],
                                label='max_samples',
                                interactive=True)
                        with gr.Row():
                            clf_params_input['bootstrap'] = gr.Checkbox(
                                value=clf_params_input['bootstrap'],
                                label='bootstrap',
                                interactive=True)
                            clf_params_input['oob_score'] = gr.Checkbox(
                                value=clf_params_input['oob_score'],
                                label='oob_score',
                                interactive=True)
                            clf_params_input['warm_start'] = gr.Checkbox(
                                value=clf_params_input['warm_start'],
                                label='warm_start',
                                interactive=True)

            with gr.Column():
                with gr.Row():
                    with gr.Accordion('检验训练集', open=False):
                        with gr.Row():
                            # 上传训练集后查看head()
                            train_display = gr.Dataframe(interactive=False)
                with gr.Row():
                    with gr.Accordion(label='算法准确度（F1_macro）'):
                        with gr.Row():
                            clf_scores_mean = gr.Label(label='平均精确度')
                        with gr.Row():
                            clf_scores = gr.Label(label='子集精确度',
                                                  num_top_classes=10)
                with gr.Row():
                    with gr.Accordion(label='下载模型文件'):
                        with gr.Row():
                            model_file = gr.Files(file_types=['.skops'],
                                                  label='下载模型文件')

    with gr.Tab('推理'):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    # 上传验证集(*.csv)
                    upload_validate_btn = gr.UploadButton(label='上传验证集(*.csv)', file_types=['.csv'])
                    # 上传测试集(*.csv)
                    upload_test_btn = gr.UploadButton(label='上传测试集(*.csv)', variant='primary', file_types=['.csv'])
                with gr.Row():
                    with gr.Accordion('检验验证集', open=False):
                        with gr.Row():
                            # 上传训练集后查看head()
                            validate_display = gr.Dataframe(interactive=False)
                with gr.Row():
                    with gr.Accordion('检验测试集', open=False):
                        with gr.Row():
                            # 上传训练集后查看head()
                            test_display = gr.Dataframe(interactive=False)
                with gr.Row():
                    with gr.Accordion('验证准确度'):
                        with gr.Row():
                            validate_scores = gr.Label(label='验证准确度')
            with gr.Column():
                with gr.Row():
                    # 上传验证集(*.csv)
                    validate_btn = gr.Button(value='验证')
                    # 上传测试集(*.csv)
                    test_btn = gr.Button(value='测试', variant='primary')
                with gr.Row():
                    with gr.Accordion(label='下载结果json文件'):
                        with gr.Row():
                            json_file = gr.Files(file_types=['.json'],
                                                 label='下载结果json文件')
                with gr.Row():
                    with gr.Accordion(label='查看结果json文件', open=False):
                        with gr.Row():
                            json_display = gr.JSON(label='查看结果json文件')

    # 触发：
    # 触发'选择模型'菜单的选项改变
    choice_name.change(fn=select_model,
                       inputs=choice_name,
                       outputs=[use_model_params,
                                clf_params_input['n_estimators'],
                                clf_params_input['criterion'],
                                clf_params_input['max_depth'],
                                clf_params_input['min_samples_split'],
                                clf_params_input['min_samples_leaf'],
                                clf_params_input['min_weight_fraction_leaf'],
                                clf_params_input['max_features'],
                                clf_params_input['max_leaf_nodes'],
                                clf_params_input['min_impurity_decrease'],
                                clf_params_input['bootstrap'],
                                clf_params_input['oob_score'],
                                clf_params_input['n_jobs'],
                                clf_params_input['random_state'],
                                clf_params_input['verbose'],
                                clf_params_input['warm_start'],
                                clf_params_input['class_weight'],
                                clf_params_input['ccp_alpha'],
                                clf_params_input['max_samples']
                                ])
    # 触发'上传模型(*.skops)'按钮
    upload_model_btn.upload(fn=model_add, inputs=upload_model_btn, outputs=choice_name)
    # 触发上传数据集按钮
    upload_train_btn.upload(fn=upload_file, inputs=upload_train_btn, outputs=train_display)
    upload_validate_btn.upload(fn=upload_file, inputs=upload_validate_btn, outputs=validate_display)
    upload_test_btn.upload(fn=upload_file, inputs=upload_test_btn, outputs=test_display)
    # 触发'数据集分割方法'
    cv_method_input.change(fn=cv_method_change,
                           inputs=cv_method_input,
                           outputs=[cv_params_input['n_splits'],
                                    cv_params_input['shuffle'],
                                    cv_params_input['random_state'],
                                    cv_params_input['p']])
    # 触发'使用模型参数'开关
    use_model_params.change(fn=clf_params_non_interactive,
                            inputs=use_model_params,
                            outputs=[clf_params_input['n_estimators'],
                                     clf_params_input['criterion'],
                                     clf_params_input['max_depth'],
                                     clf_params_input['min_samples_split'],
                                     clf_params_input['min_samples_leaf'],
                                     clf_params_input['min_weight_fraction_leaf'],
                                     clf_params_input['max_features'],
                                     clf_params_input['max_leaf_nodes'],
                                     clf_params_input['min_impurity_decrease'],
                                     clf_params_input['bootstrap'],
                                     clf_params_input['oob_score'],
                                     clf_params_input['n_jobs'],
                                     clf_params_input['random_state'],
                                     clf_params_input['verbose'],
                                     clf_params_input['warm_start'],
                                     clf_params_input['class_weight'],
                                     clf_params_input['ccp_alpha'],
                                     clf_params_input['max_samples']])
    # 触发'训练'按钮
    train_btn.click(fn=training,
                    inputs=[upload_train_btn,
                            cv_method_input,
                            cv_params_input['n_splits'],
                            cv_params_input['shuffle'],
                            cv_params_input['random_state'],
                            cv_params_input['p'],
                            use_model_params,
                            clf_params_input['n_estimators'],
                            clf_params_input['criterion'],
                            clf_params_input['max_depth'],
                            clf_params_input['min_samples_split'],
                            clf_params_input['min_samples_leaf'],
                            clf_params_input['min_weight_fraction_leaf'],
                            clf_params_input['max_features'],
                            clf_params_input['max_leaf_nodes'],
                            clf_params_input['min_impurity_decrease'],
                            clf_params_input['bootstrap'],
                            clf_params_input['oob_score'],
                            clf_params_input['n_jobs'],
                            clf_params_input['random_state'],
                            clf_params_input['verbose'],
                            clf_params_input['warm_start'],
                            clf_params_input['class_weight'],
                            clf_params_input['ccp_alpha'],
                            clf_params_input['max_samples']],
                    outputs=[clf_scores_mean, clf_scores, model_file])
    validate_btn.click(fn=validate,
                       inputs=upload_validate_btn,
                       outputs=[validate_scores,
                                json_file,
                                json_display])
    test_btn.click(fn=test,
                   inputs=upload_test_btn,
                   outputs=[json_file,
                            json_display])

if __name__ == '__main__':
    # demo.queue()
    demo.launch()
