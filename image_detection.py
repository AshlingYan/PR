from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import torch

def detect_objects(image_path, query0, query1):
    
    # 初始化处理器和模型
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    # 读取图片
    image = Image.open(image_path)

    # 组装文本查询列表
    text_queries = [query0, query1]

    # 准备输入数据
    inputs = processor(text=text_queries, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # 设置目标图像大小以重新缩放框预测
    target_sizes = torch.Tensor([image.size[::-1]])

    # 将输出（边界框和类 logits）转换为Pascal VOC格式
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)

    # 处理每张图像的结果
    for i, result in enumerate(results):
        boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
        detected = [False] * len(text_queries)

        for box, score, label in zip(boxes, scores, labels):
            if score.item() >= 0.1 and label < len(text_queries):  # 确保标签索引不会越界
                detected[label] = True
                x_center = (box[0].item() + box[2].item()) / 2
                y_center = (box[1].item() + box[3].item()) / 2
                x_center, y_center = round(x_center, 2), round(y_center, 2)
                print(f"Detected {text_queries[label]} with confidence {round(score.item(), 3)} at center location ({x_center}, {y_center})")

        # 检查哪些文本查询没有检测到任何符合阈值的对象
        for j, was_detected in enumerate(detected):
            if not was_detected:
                print(f"对于‘{text_queries[j]}’，没有检测到符合的对象。")
