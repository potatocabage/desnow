from configs import open_configs
from load_data import load_data
from format_data import format_data
from trainer import Trainer
from time import time
import resource
import sys



def pipeline():
    data_configs, training_configs, model_configs = open_configs()

    train_datasets,\
    val_datasets,\
    test_l_datasets,\
    test_m_datasets,\
    test_s_datasets = load_data(data_configs)

    train_dataloader,\
    val_dataloader,\
    test_l_dataloader,\
    test_m_dataloader,\
    test_s_dataloader = format_data(train_datasets,
                                     val_datasets,
                                     test_l_datasets,
                                     test_m_datasets,
                                     test_s_datasets,
                                     training_configs)

    print('start trainer')
    trainer = Trainer(training_configs, model_configs, data_configs)
    best_model, history, test_l_loss, test_m_loss, test_s_loss = trainer.train_val_test(train_dataloader,
                                                                                        val_dataloader,
                                                                                        test_l_dataloader,
                                                                                        test_m_dataloader,
                                                                                        test_s_dataloader)
    print('done trainer')
    return history



def memory_limit():
    """Limit max memory usage to half."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # Convert KiB to bytes, and divide in two to half
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 // 2, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory  # KiB

if __name__ == '__main__':
    memory_limit()
    try:
        pipeline()
        print("Done!")
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)

# if __name__ == "__main__":
#     pipeline()
#     print("Done!")


