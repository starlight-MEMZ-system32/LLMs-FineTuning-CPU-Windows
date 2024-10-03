import os
import sys
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

print("请确认你的模型文件位于当前程序工作目录下出现的model文件夹中（输入任意字符后回车继续）")

os.makedirs('model', exist_ok=True)

# 暂停程序
input()

print("载入模型......")

# 模型路径
model_path = "./model"

try:
    # 载入模型
    model = AutoModelForCausalLM.from_pretrained(model_path)
except OSError:
    print("未找到模型文件，请确定模型文件完整且路径正确！程序将退出......")
    sys.exit()

print("完成！")

print("请确定模型的分词器文件（vocab.json）位于model文件夹中（输入任意字符后回车继续）")

# 暂停程序
input()

print("加载分词器......")


# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only = True,  # 只从本地文件系统中加载
    vocab_file = './model/vocab.json'  # 从本地加载分词器
)

print("完成！")

device = input("请选择模型训练使用的设备，输入C为CPU，输入N为CUDA单元（需要N卡，如果是在Linux环境下也可调用A卡），输入I为XMX单元（需要I卡，暂未实现）")


# 选择训练设备
if device == "C":
    print("将在CPU上训练模型！")
    model.to("cpu")
elif device == "N":
    print("将在显卡上训练模型！（未经测试，可能无法正常工作）")
    try:
        model.to("cuda")
    except AssertionError:
        print("出现异常！你的显卡可能不支持训练AI或环境设置有误，输入任意字符后回车使用CPU继续训练...")
        input()
        model.to("cpu")
elif device == "I":
    print("在做了在做了qwq 现在先用着CPU吧QwQ")
    model.to("cpu")
else:
    print("非法输入！将使用CPU继续训练...")
    model.to("cpu")

train_data_name = input("请确认训练集文件位于工作目录后输入训练集文件的名称（需要带上扩展名）")

print("由于本人技术不佳，暂时只支持输入和输出两个值，如有需要可自己在代码中增加对应字典中的字段")

user_input = input("请输入训练集中代表用户输入的键名（如input）")
ai_output = input("请输入训练集中代表AI输出的键名（如output）")

# 加载 JSON 数据集
def load_json_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    # 将数据转换为字典格式（如果需要更多输入就是改这里）
    dataset_dict = {user_input: [item[user_input] for item in data], ai_output: [item[ai_output] for item in data]}
    return dataset_dict
#try:
train_data = load_json_dataset(train_data_name)
#except:

# 转换为 Hugging Face Dataset 对象
train_dataset = Dataset.from_dict(train_data)

# 数据预处理
def preprocess_function(examples):
    inputs = examples[user_input]
    outputs = examples[ai_output]
    # 对输入和输出进行分词
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(outputs, max_length=512, truncation=True, padding="max_length")
    # 确保输入和标签的长度相同
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)


def get_training_arguments():
    # 默认训练参数
    default_args = {
        "num_train_epochs": 10,  # 指定训练周期的数量，每个周期都会遍历一次数据集
        "per_device_train_batch_size": 4,  # 指定训练批次大小，根据CPU与内存大小进行调整
        "gradient_accumulation_steps": 4,  # 梯度累积的步数，如果内存较小，可以通过增加梯度累积步数来模拟更大的批次大小
        "learning_rate": 0.00002,  # 学习率，控制模型训练过程中参数更新的速度
        "logging_steps": 10,  # 指定日志记录的间隔步数，每10步记录一次训练信息
    }

    # 获取用户输入的训练参数
    try:
        num_train_epochs = int(input("请输入训练周期的数量（默认:10）: ") or int(default_args["num_train_epochs"]))
    except ValueError:
        num_train_epochs = default_args["num_train_epochs"]
    try:
        per_device_train_batch_size = int(input("请输入训练批次大小（默认:4）: ") or int(default_args["per_device_train_batch_size"]))
    except ValueError:
        per_device_train_batch_size = default_args["per_device_train_batch_size"]
    try:
        gradient_accumulation_steps = int(input("请输入梯度累积步数（默认:4）: ") or int(default_args["gradient_accumulation_steps"]))
    except ValueError:
        gradient_accumulation_steps = default_args["gradient_accumulation_steps"]
    try:
        learning_rate = float(input("请输入学习率（默认:0.00002）: ") or float(default_args["learning_rate"]))
    except ValueError:
        learning_rate = default_args["learning_rate"]
    try:
        logging_steps = int(input("请输入日志记录的间隔步数（默认:10）: ") or int(default_args["logging_steps"]))
    except ValueError:
        logging_steps = default_args["logging_steps"]

    print("数据导入中......")

    # 创建训练参数对象
    training_args = TrainingArguments(
        output_dir = "./results", # 指定模型训练输出的目录，所有训练结果和模型文件将保存在这里
        num_train_epochs = num_train_epochs,
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        learning_rate = learning_rate,
        logging_dir = "./logs",
        logging_steps = logging_steps,
    )
    return training_args

# 调用函数获取训练参数
training_args = get_training_arguments()

print("创建 Trainer 对象......")

# 创建 Trainer 对象
trainer = Trainer(
    model = model  ,
    args = training_args ,
    train_dataset = tokenized_train_dataset ,
    # eval_dataset=tokenized_validation_dataset,
    tokenizer = tokenizer,
)

print("开始训练......")

# 开始微调
trainer.train()

print("完成！训练好的模型已经保存到工作目录下的finetuned_model文件夹！")

# 保存模型
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")
