import shutil

if __name__ == '__main__':
    src_file = "/Users/liushun/Documents/毕业论文/code/go/mev_auction/bsc-rpc-client/log/rpc.log"
    dst_file = "./log_files/bsc-rpc-client.log"
    shutil.copy(src_file, dst_file)

    src_file = "/Users/liushun/Documents/毕业论文/code/go/mev_auction/bsc-rpc/log/rpc.log"
    dst_file = "./log_files/bsc-rpc.log"
    shutil.copy(src_file, dst_file)
