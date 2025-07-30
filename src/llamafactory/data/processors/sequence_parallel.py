from ...extras.constants import IGNORE_INDEX
from ..data_utils import preprocess_sp_dataset


def get_max_lengths(examples):
    valid_lists = []
    for key, value in examples.items():
        if key.endswith('input_ids') and value is not None:
            valid_lists.append(value)
    
    if not valid_lists:
        return []
    
    max_lengths = [max(len(lst) if lst is not None else 0 for lst in group) 
                   for group in zip(*valid_lists)]
    
    return max_lengths


def pad_sequence(examples, data_args, tokenizer, model_args):
    max_length = data_args.cutoff_len
    input_pad_token_id = tokenizer.pad_token_id
    assert data_args.ignore_pad_token_for_loss
    label_pad_token_id = IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    max_input_ids_length_list = get_max_lengths(examples)
    for k, v in examples.items():
        if k.endswith("input_ids"):
            pad_token_id = input_pad_token_id
        elif k.endswith("labels"):
            pad_token_id = label_pad_token_id
            # shift labels here
            for i in range(len(v)):
                v[i] = v[i][1:]
        elif k.endswith("attention_mask"):
            pad_token_id = 0
        elif k.endswith("position_ids"):
            pad_token_id = max_length - 1  # pad the max position id
        elif k == "images" or k == "videos":
            pad_token_id = -1
            continue  # TODO: haven't tested multi-modal yet
        else:
            raise NotImplementedError(f"Unexpected dataset key: {k}")

        for i in range(len(v)):
             tmp_sp_len = max_input_ids_length_list[i] // model_args.sequence_parallel_size
             closest_cutoff_len = int(tmp_sp_len + (8 - tmp_sp_len % 8)) * model_args.sequence_parallel_size
             max_length = min(closest_cutoff_len, data_args.cutoff_len)

             v[i].extend([pad_token_id] * (max_length - len(v[i])))
        examples[k] = v

    return examples


def create_image_position_info(seq_ids, image_token_id):
    """为整个序列创建图像位置信息"""
    info = []
    global_image_pos = 0  # 全局连续的图像位置计数器
    
    for token_id in seq_ids:
        if token_id == image_token_id:
            info.append(global_image_pos)
            global_image_pos += 1
        else:
            info.append(-1)
    return info


# sp for Sequence Parallel
def sp_split(examples, model_args, tokenizer):
    all_image_position_maps = list()
    new_examples = dict()

    for k, v in examples.items():
        chunks = list()
        for row in v:
            #if k.endswith("attention_mask"):
            #    chunks.extend([row] * model_args.sequence_parallel_size)
            if row is None:
                chunks.extend([None] * model_args.sequence_parallel_size)
            elif k in ['images']:
                chunks.extend([row] * model_args.sequence_parallel_size)
            else:
                chunks.extend(
                    preprocess_sp_dataset(row, model_args.sequence_parallel_size, model_args.sequence_parallel_mode)
                )
                if k.endswith("input_ids") and len(all_image_position_maps) < (len(v) * model_args.sequence_parallel_size):
                    image_position_info = create_image_position_info(row, tokenizer.image_token_id)
                    all_image_position_maps.extend(
                        preprocess_sp_dataset(image_position_info, model_args.sequence_parallel_size, model_args.sequence_parallel_mode)
                    )
        new_examples[k] = chunks
    
    if len(all_image_position_maps)>0:
        new_examples['image_position_maps'] = all_image_position_maps
        for index in range(len(new_examples['images'])):
            if all(image_position==-1 for image_position in new_examples['image_position_maps'][index]):
                new_examples['images'][index] = None

    return new_examples 
