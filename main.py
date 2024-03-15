from ultralytics import YOLO

# 安装命令
# python setup.py develop

# 数据集示例百度云链接
# 链接：https://pan.baidu.com/s/19FM7XnKEFC83vpiRdtNA8A?pwd=n93i
# 提取码：n93i

if __name__ == '__main__':
    # 直接使用预训练模型创建模型.
    # model = YOLO('yolov8x.pt')
    # model.train(**{'cfg': 'ultralytics/yolo/cfg/default.yaml', 'data': 'ultralytics/datasets/coco.yaml'})

    # 使用yaml配置文件来创建模型,并导入预训练权重.
    # model = YOLO('/ultralytics/cfg/models/v8/yolov8x-SPPFImproveAndFAM.yaml')
    # model.load('/ultrayltics-main/yolov8x.pt')
    # model.train(**{'cfg': 'ultralytics/yolo/cfg/default.yaml', 'data': '/ultrayltics-main/ultralytics/cfg/datasets/coco.yaml'})
    
    # 模型验证
    model = YOLO('/home/white/sharedirs/两个创新一起.pt')
    model.val(**{'data': '/root/autodl-tmp/ultrayltics-main/ultralytics/cfg/datasets/coco.yaml','split':'test'})
    
    # # 模型推理
    # model = YOLO('runs/detect/yolov8n_exp/weights/best.pt')
    # model.predict(source='dataset/images/test', **{'save': True})
    
#     #断点续训
#     # Load a model
#     model = YOLO('/root/autodl-tmp/ultrayltics-main/runs/detect/train5/weights/last.pt')  # load a partially trained model

#     # Resume training
#     results = model.train(resume=True)