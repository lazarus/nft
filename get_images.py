import pandas as pd
import wget
import multiprocessing

def run_process(url, output_path):
    wget.download(url, out=output_path)

if __name__ == '__main__':
    cpus = multiprocessing.cpu_count()
    max_pool_size = 4
    pool = multiprocessing.Pool(cpus if cpus < max_pool_size else max_pool_size)

    df = pd.read_json("set_10000.json")

    train_images = df[['image_url','token_id']]
    print(f"Please wait, downloading {train_images.index.size} images.")
    for ind in train_images.index:
        image_url = df['image_url'][ind]
        filename = df['token_id'][ind]
        
        pool.apply_async(run_process, args=(f"{image_url}=s128", f"out/{filename}.png", ))

    pool.close()
    pool.join()
    print("Finished")