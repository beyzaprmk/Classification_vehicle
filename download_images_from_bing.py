from bing_images import bing

bing.download_images("広告用カーラッピング",
                      100,
                      output_dir=r"C:\Users\beyza.parmak\Desktop\data",
                      pool_size=10,
                      #file_type="png",
                      force_replace=False,
                      extra_query_params='&first=1')