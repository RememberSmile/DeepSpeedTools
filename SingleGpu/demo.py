import json
import torch
import math
from utils import DeepspeedStrategy
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.trainer import get_scheduler

with open("ds_config.json","r") as f:
    ds_config = json.load(f)
  

# 构造deepspeed策略
strategy = DeepspeedStrategy(
        seed = ds_config["seed"],
        max_norm = ds_config["max_norm"],
        micro_train_batch_size = ds_config["micro_train_batch_size"],
        train_batch_size = ds_config["train_batch_size"],
        zero_stage = ds_config["zero_stage"],
        bf16 = ds_config["bf16"],
        use_adam_offload = ds_config["adam_offload"]
    )

# 加载模型
model_name_or_path = "Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
model =  LlamaForCausalLM.from_pretrained(model_name_or_path)
#model = LlamaForCausalLM.from_pretrained(model_name_or_path,device_map=tmp)

# move gpu
model.cuda()

# 创建优化器
optim = strategy.create_optimizer(model, lr=0.1, betas=(0.9, 0.95),weight_decay=0.0)

# 构建数据集
# 创建DeepSpeed配置
def create_dataloader():
    # 创建示例数据集和数据加载器。这里只是一个示例，你需要根据实际情况编写
    inputs = tokenizer("DeepSpeed is awesome!", return_tensors="pt")
    labels = tokenizer("DeepSpeed 是很棒的!", return_tensors="pt")
    dataset = torch.utils.data.TensorDataset(inputs['input_ids'], labels['input_ids'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    return dataloader

dataloader = create_dataloader()

max_steps = 1000
scheduler = get_scheduler(
        "cosine",
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
    )

(model, optim, scheduler) = strategy.prepare((model, optim, scheduler))



# 训练模型
model.train()
# # 尝试冻结权重
# for n, p in model.named_parameters():
#     p.requires_grad = False

for epoch in range(200):
    for batch in dataloader:
        input_batch = batch[0]
        chosen_ids = input_batch.squeeze(1).to(torch.cuda.current_device())
        print("chosen_ids:",chosen_ids)
        # # 传播之前
        # model.module.model.embed_tokens.weight[1,:] = model.module.model.embed_tokens.weight[1,:].detach()
        # model.module.model.embed_tokens.weight[21784,:] = model.module.model.embed_tokens.weight[21784,:].detach()
        # model.module.model.embed_tokens.weight[26539,:] = model.module.model.embed_tokens.weight[26539,:].detach()
        
        model_output = model(chosen_ids, labels=chosen_ids)
        loss = model_output["loss"]
        print("loss:",loss)

        print("############################################################")
        print(model.module.model.embed_tokens.weight[1,:].grad())
        print(model.module.model.embed_tokens.weight[21784,:].grad())
        print(model.module.model.embed_tokens.weight[26539,:].grad())
        print("############################################################")

        print("#####backward#####")
        strategy.backward(loss, model, optim)
        # print("############ model ############")
        # print(model)
        # print("-------------modules-----------------")
        # print(model.modules)
        # print("*************model.modules.model*****************")
        # print(model.module.model.embed_tokens.weight)
        print("------------------------------------------------------------")
        print("**********1 model.modules.model.weight *********")
        print(model.module.model.embed_tokens.weight[1,:])
        print("**********21784 model.modules.model.weight *********")
        print(model.module.model.embed_tokens.weight[21784,:])
        print("**********26539 model.modules.model.weight *********")
        print(model.module.model.embed_tokens.weight[26539,:])
        
        print("**********338 model.modules.model.weight *********")
        print(model.module.model.embed_tokens.weight[338,:])
        print("**********29663 model.modules.model.weight *********")
        print(model.module.model.embed_tokens.weight[29663,:])
        print("**********29991 model.modules.model.weight *********")
        print(model.module.model.embed_tokens.weight[29991,:])
        print("------------------------------------------------------------")

        # print("######### model.modules.model.weight ##########")
        # print(model.module.model.embed_tokens.weight.grad[21784])

        #print(model.module.model.embed_tokens.weight.grad)
        
        # print("************model grad zero***********************")
        # # 扣掉其他所有token的梯度的梯度,只保留新增token对应embedding得题目
        # tokens_to_freeze = [i for i in range(0,32000)]
        # for token in tokens_to_freeze:
        #     model.module.model.embed_tokens.weight.grad[token] = 0   
        #exit()

        #model.model.embed_tokens.weight.grad[100] = 0 
        strategy.optimizer_step(optim, model, scheduler)