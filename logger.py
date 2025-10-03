import sys
from loguru import logger
import hashlib
from datetime import datetime



# log_file = os.path.join(log_dir, 'logs.txt')

info_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "  
    "<level>{level: <8}</level> | "                
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - " 
    "<fg #ACE1AF>{message}</fg #ACE1AF>"
)
logger.remove() 

def init_logger(args):
    logger.remove()

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    
    log_name = f'[{args.stage}]_[{args.model}]_[trial={args.trial_id}]_{current_time}'
    md5 = hashlib.md5(log_name.encode(encoding="utf-8")).hexdigest()[:6]
    log_name = log_name+'_' + md5
    args.md5 = md5
    args.current_time = current_time
    args.file_log_dir = args.log_dir+"/" + log_name +".log"
    logger.add(
        args.file_log_dir, 
        format=info_format, level="INFO",enqueue=True)
    logger.add(sys.stdout, format=info_format, level="INFO",enqueue=True)
    
    logger.info('\n'.join([str(k) + ': ' + str(v) for k, v in vars(args).items()]),)
    
if __name__ == '__main__':

    class Args:
        def __init__(self):
            self.stage = 'train'
            self.model = 'CDSASRec'
            self.trial_id = 1
            self.md5 = None
    args = Args()
    init_logger('./', args)
        