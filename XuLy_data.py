import os
import random
from PIL import Image
import glob
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import shutil

if __name__ == "__main__":
    # check thông tin data
    data_path = "C:/pythonProject/Pycharm_pythoncode/garbage_classification"
    count_all_data = []
    for subdir in os.listdir(data_path):
        subdir_path = os.path.join(data_path, subdir)
        # Pastikan hanya memproses folder
        if os.path.isdir(subdir_path):
            image_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            num_images = len(image_files)
            count_all_data.append(num_images)
            print(f"{subdir}: {num_images}")
    print(f"\nTotal: {sum(count_all_data)}")
    # ------------------------------------------------------------------
    # battery: 945
    # biological: 985
    # brown-glass: 607
    # cardboard: 891
    # clothes: 5325 # cần pahir giảm hình ảnh để tránh bị imbelance data
    # green-glass: 629
    # metal: 769
    # paper: 1050
    # plastic: 865
    # shoes: 1977
    # trash: 697
    # white-glass: 775
    #
    # Total: 15515
    # ------------------------------------------------------------------
    # chỉ để lại 5 thư mục thừ 12 thư mục
    new_data_path = "C:/pythonProject/Pycharm_pythoncode/garbage_classification_merge"
    category_mapping = {
        "battery": "non-recyclable",
        "biological": "non-recyclable",
        "trash": "non-recyclable",
        "brown-glass": "glass",
        "green-glass": "glass",
        "white-glass": "glass",
        "cardboard": "paper",
        "paper": "paper",
        "clothes": "fabric",
        "shoes": "fabric",
        "metal": "recyclable-inorganic",
        "plastic": "recyclable-inorganic"
    }
    # đổi tên, tạo thư mục cha con
    new_data_path = os.path.join(os.path.dirname(data_path),'garbage_classification_merge')
    os.makedirs(new_data_path, exist_ok=True)
    # lấy các nhóm ở trên để làm thư mục con của thư mục cha
    for new_category in set(category_mapping.values()):
        new_category_path = os.path.join(new_data_path,new_category)
        os.makedirs(new_category_path, exist_ok=True)
    # tạo khóa duy nhất cho mỗi value
    file_counters = {key: 1 for key in category_mapping.values()} # cái này để tí đánh số cho các category
    # giới hạn mục clothes trong khoảng 2000, do clothes chứa nhiều ảnh cho nên cần cắt giảm để tránh imbelace data
    clothes_path = os.path.join(data_path, "clothes")
    if os.path.exists(clothes_path):
        clothes_file = os.listdir(clothes_path)
        if len(clothes_file) > 2000:
            clothes_file = random.sample(clothes_file, 2000)
    # sao chép ảnh từ danh mục cũ
    # tạo ra 2 đường dẫn đến các thư mục con trong thư mục cha, old_cate_path là cũ trong đó có chứa ảnh, new_cate_path là mới trong đó chưa có gì
    for old_cate, new_cate in category_mapping.items():
        old_cate_path = os.path.join(data_path, old_cate)
        new_cate_path = os.path.join(new_data_path,new_cate)
        # nếu tồn tại thì lấy các ảnh ở trong file đó ra lưu vào file_list, nếu là ảnh của clothes thì dùng cách lấy ở trên
        if os.path.exists(old_cate_path):
            file_list = os.listdir(old_cate_path)
            if old_cate == "clothes":
                file_list = clothes_file #sử dụng danh sách đã giảm
            # lấy từng ảnh ra, duyệt từng ảnh và sao chép vào thư mục mới ví dụ ở đây đang duyệt: old_file_path = "/datasets/garbage-classification/clothes/cloth1.jpg"
            for file_name in file_list:
                old_file_path = os.path.join(old_cate_path, file_name)
                # Kiểm tra và đổi tên file trước khi sao chép
                if os.path.isfile(old_file_path): # chỉ sao chép file không sao chép mục, duyệt từ file ảnh và chuyển ảnh thành dạng jpg
                    new_file_name = f"{new_cate}{file_counters[new_cate]}.jpg"
                    file_counters[new_cate] +=1 # cập nhật ảnh theo thứ tự ví dụ non-recyclable1.jpg, non-recyclable2.jpg
                    #Sao chép ảnh vào thư mục mới ví dụ:
                    #shutil.copy2("/datasets/garbage-classification/clothes/cloth1.jpg","/datasets/garbage-classification-merged/fabric/fabric1.jpg")
                    new_file_path = os.path.join(new_cate_path, new_file_name)
                    shutil.copy2(old_file_path, new_file_path)
            print(f"Done quá trình chuyển đổi từ '{old_cate}' sang '{new_cate}'.")
    print("Done!")
    count_all_data_1 = []
    for subdir in os.listdir(new_data_path):
        subdir_path = os.path.join(new_data_path, subdir)
        if os.path.isdir(subdir_path):
            image_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            num_images = len(image_files)
            count_all_data_1.append(num_images)
            print(f"{subdir}: {num_images}")
    # fabric: 3977
    # glass: 2011
    # non-recyclable: 2627
    # paper: 1941
    # recyclable-inorganic: 1634
    # kiểm tra độ phân giải của nhóm ảnh trong đó check the resolution
    def print_image_size(dictionary, max_image_per_class=10):
        total_image = 0
        for subdir in os.listdir(dictionary):
            subdir_path = os.path.join(dictionary,subdir)
            if not os.path.isdir(subdir_path):
                continue
            image_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png','.jpg','.jpeg'))]
            sampled_files = image_files[:max_image_per_class]
            num_images = len(sampled_files)
            print(f"{subdir}: {num_images}")
            total_image += num_images
            unique_size = set()
            for img_file in sampled_files:
                img_path = os.path.join(subdir_path,img_file)
                with Image.open(img_path) as img:
                    unique_size.add(img.size)
            for size in unique_size:
                print(f"-{size}")
            print("----------------")
    print_image_size(new_data_path, max_image_per_class=35)
    """
    bây giờ chúng ta đã có 1 thư mục chứa các label và hiểu được kích thước bức ảnh
    giống như machine learning ta sẽ sáo trộn bức ảnh ngẫu nhiên để tránh mô hình học tuần tự sẽ gặp vấn đề về bias
    sau đó ta tạo thành 1 dataframe chứa image và label để dùng train_test_split chia ra làm 2 tập train và test 
    """
    print(pd.DataFrame(os.listdir(new_data_path), columns=['Folder_Name']))
    #             Folder_Name
    # 0                fabric
    # 1                 glass
    # 2        non-recyclable
    # 3                 paper
    # 4  recyclable-inorganic
    # tạo ra 1 list danh sách các đường dẫn tới ảnh
    files = [i.replace("\\", "/") for i in glob.glob(new_data_path + "/*/*")]
    # # xáo trộn lung ta lung tung
    np.random.shuffle(files)# trộn các bức ảnh ngẫu nhiên
    # # lấy đường dẫn thư mục chứa file và lấy tên thư mục cuối cùng
    labels = [os.path.dirname(i).split('/')[-1] for i in files]
    data = zip(files, labels)
    dataframe = pd.DataFrame(data, columns=['Image', "Label"])
    lists = dataframe['Image'].head(5)
    for i in lists:
        image = cv2.imread(i)
        cv2.imshow("ảnh", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    '''
    bây giờ từ những tấm ảnh đã có ta cần tạo ra 1 file train, test, value 
    '''
    # tạo thư mục cho train test và value
    mainPath = "C:/pythonProject/Pycharm_pythoncode"
    train_dir = mainPath + "/Dataset/train"
    val_dir = mainPath + "/Dataset/val"
    test_dir = mainPath + "/Dataset/test"

    # lấy ra các thư mục con của thư mục cha rồi đánh số cho từng labels
    classes = os.listdir(new_data_path)  # la 1 list
    classes_indices = {name: index for index, name in enumerate(classes)}
    # print(classes_indices)
    for directory in [train_dir, val_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
        for cls in classes:
            os.makedirs(os.path.join(directory, cls), exist_ok=True)

    # giữ nguyên quá trình phân chia
    random.seed(42)
    """
    Tạo hàm phân chia dữ liệu nhiều tầng: hay
    stratify ở đây tác gải đã chia ra làm 2 trường hợp nếu như số lượng bức ảnh> 1 người ta sẽ chia ảnh theo ý hiểu sau:
    ban đầu tỉ lệ các nhãn trong bức ảnh là : 50%A, 30%B,20%C thì sau khi chia tập train lấy 90% tổng số ảnh nhưng sẽ vẫn giữ tỉ lệ 5-3-2 với mỗi labels
    điều này tránh mất cân bằng dữ liệu khi ở trên ta đã xử lý các nhãn có số lượng ảnh gần tương đương nhau.
    Nếu như chỉ có 1 nhãn người ta sẽ dùng shuffle để xáo trộn ngẫu nhiên.
    """

    def safe_train_test_split(images, test_size, stratify_labels=None):
        if stratify_labels is not None and len(set(stratify_labels)) > 1:
            return train_test_split(images, test_size=test_size, stratify=stratify_labels, random_state=42)
        else:
            return train_test_split(images, test_size=test_size, random_state=42, shuffle=True)
        # lặp qua từng lớp ảnh
    for cls in classes:
        class_path = os.path.join(new_data_path, cls).replace("\\", "/")
        if os.path.isdir(class_path):
            images = os.listdir(class_path)  # lúc này images sẽ chứa đường dẫn của các ảnh.
            # bỏ qua các lớp có ít ảnh
            if len(images) < 2:
                print(f"Skip '{cls}', vì số lượng ảnh là {len(images)} không đủ để phân chia")
                continue
            # tạo nhãn để phân tầng dữ liệu, trong đó tất cả các ảnh của 1 lớp đều cùng 1 nhãn
            '''
            stratify sẽ ấn định số ảnh khi cho vào mục, người ta nhân để tạo đủ nhãn cho số lượng ảnh, khi mà lấy 90% bức ảnh cho train, 10% còn lại là 
            temp_images, xong lại lấy cái đó là chỉ số để ấn định xem nào có bao nhiêu bức ảnh được lấy ra ở mỗi mục, và cũng từ đó chia 10% thành val và test
            '''
            labels = [cls] * len(images)
            train_images, temp_images = safe_train_test_split(images, test_size=0.10,
                                                              stratify_labels=labels)  # dùng cái này để đảm bảo số lượng ảnh của mỗi thưu mục con được chia đúng tỉ lệ
            val_images, test_images = safe_train_test_split(temp_images, test_size=0.30,
                                                            stratify_labels=[cls] * len(temp_images))

            os.makedirs(os.path.join(train_dir, cls).replace("\\", "/"), exist_ok=True)
            os.makedirs(os.path.join(val_dir, cls).replace("\\", "/"), exist_ok=True)
            os.makedirs(os.path.join(test_dir, cls).replace("\\", "/"), exist_ok=True)

            '''
            Hàm này sẽ làm công việc chuyển đổi ảnh từ tập nguồn sang tập tạo, image_list là thư mục vừa được phân chia,
            class_path là đường dẫn tới thư mục chứa ảnh, destination là đường dẫn tới thư mục ảnh của thư mục mới tạo
            Mô tả: đầu tiên nếu trong thư mục mới tạo như train_dir, val_dir chưa có ảnh nào, ảnh sẽ được lấy từ train_image tức là thư
            mục vừa được phân chia ảnh ở trên sao chép ảnh từ tập nguồn đến tập đích: source là đường dẫn đến thư mục ảnh gốc + image là ảnh
            copy và chuyển sang destination là đường dẫn đến thư mục ảnh mới tạo + image là ảnh.

            '''


            def move_images(image_list, source, destination):
                if len(os.listdir(destination)) == 0:
                    for image in image_list:
                        shutil.copy(os.path.join(source, image).replace("\\", "/"),
                                    os.path.join(destination, image).replace("\\", "/"))
                    # print(f"Thư mục đích là: '{destination}' số lượng ảnh là: [{len(image_list)}]")
                else:
                    # print(f" Ảnh đã có trong: '{destination}', không sao chép để tránh trùng lặp.")
                    print("")

            move_images(train_images, class_path, os.path.join(train_dir, cls).replace("\\", "/"))
            move_images(val_images, class_path, os.path.join(val_dir, cls).replace("\\", "/"))
            move_images(test_images, class_path, os.path.join(test_dir, cls).replace("\\", "/"))
            # print("-" * 60)
    """
    Hai điều cần khắc phục ở đây 1 là thay \ bằng /, hai là fabric đang chiếm quá nhiều ta cần loại bỏ bớt ảnh ở đó đi để
    máy học cân bằng hơn. sử dụng .replace("\\", "/") vì os.path.join trên windown sẽ là dấu \
    Thư mục đích là: 'C:/pythonProject/Pycharm_pythoncode/Dataset/train\fabric' số lượng ảnh là: [3579]
    Thư mục đích là: 'C:/pythonProject/Pycharm_pythoncode/Dataset/val\fabric' số lượng ảnh là: [278]
    Thư mục đích là: 'C:/pythonProject/Pycharm_pythoncode/Dataset/test\fabric' số lượng ảnh là: [120]
    ------------------------------------------------------------
    Thư mục đích là: 'C:/pythonProject/Pycharm_pythoncode/Dataset/train\glass' số lượng ảnh là: [1809]
    Thư mục đích là: 'C:/pythonProject/Pycharm_pythoncode/Dataset/val\glass' số lượng ảnh là: [141]
    Thư mục đích là: 'C:/pythonProject/Pycharm_pythoncode/Dataset/test\glass' số lượng ảnh là: [61]
    ------------------------------------------------------------
    Thư mục đích là: 'C:/pythonProject/Pycharm_pythoncode/Dataset/train\non-recyclable' số lượng ảnh là: [2364]
    Thư mục đích là: 'C:/pythonProject/Pycharm_pythoncode/Dataset/val\non-recyclable' số lượng ảnh là: [184]
    Thư mục đích là: 'C:/pythonProject/Pycharm_pythoncode/Dataset/test\non-recyclable' số lượng ảnh là: [79]
    ------------------------------------------------------------
    Thư mục đích là: 'C:/pythonProject/Pycharm_pythoncode/Dataset/train\paper' số lượng ảnh là: [1746]
    Thư mục đích là: 'C:/pythonProject/Pycharm_pythoncode/Dataset/val\paper' số lượng ảnh là: [136]
    Thư mục đích là: 'C:/pythonProject/Pycharm_pythoncode/Dataset/test\paper' số lượng ảnh là: [59]
    ------------------------------------------------------------
    Thư mục đích là: 'C:/pythonProject/Pycharm_pythoncode/Dataset/train\recyclable-inorganic' số lượng ảnh là: [1470]
    Thư mục đích là: 'C:/pythonProject/Pycharm_pythoncode/Dataset/val\recyclable-inorganic' số lượng ảnh là: [114]
    Thư mục đích là: 'C:/pythonProject/Pycharm_pythoncode/Dataset/test\recyclable-inorganic' số lượng ảnh là: [50]
    """

    '''
    Đảm bảo rằng tên mỗi ảnh chỉ xuất hiện trong đúng 1 thư mục không có chuyện ảnh này xuất hiện ở train nhưng cũng xuất hiện
    ở test
    '''
    # khởi tạo tập hợp để lưu danh sách ảnh ở mỗi thư mục

    train_images = set()
    val_images = set()
    test_images = set()
    for cls in os.listdir(train_dir):
        train_images.update(os.listdir(os.path.join(train_dir,cls).replace("\\","/")))# update để thêm nhiều phần tử vào cùng 1 lúc thay vì add thường phần tử 1
    for cls in os.listdir(val_dir):
        val_images.update(os.listdir(os.path.join(val_dir, cls).replace("\\","/")))
    for cls in os.listdir(test_dir):
        test_images.update(os.listdir(os.path.join(test_dir, cls).replace("\\","/")))

    overlap = train_images & val_images | train_images & test_images | val_images & test_images# tìm ảnh trùng lặp dữ liệu
    if overlap:
        print(f"Duplicate images found: {overlap}")
    else:
        print("No duplicate images found.")
    # No duplicate images found.

    '''
    in ra số lượng ảnh trong mỗi tập dữ liệu
    '''
    def count_images_in_subfolders(directory):
        total_images = 0
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):  # Chỉ đếm thư mục
                total_images += len(os.listdir(subdir_path))  # Đếm số ảnh trong subfolder
        return total_images
    train_samples = count_images_in_subfolders(train_dir)
    validation_samples = count_images_in_subfolders(val_dir)
    test_samples = count_images_in_subfolders(test_dir)
    print(f"Train samples: {train_samples}")
    print(f"Validation samples: {validation_samples}")
    print(f"Test samples: {test_samples}\n")

    print(f"Total data: {train_samples + validation_samples + test_samples}")
    print(f"Total classes: {len(classes)}")
    # Train samples: 10968
    # Validation samples: 853
    # Test samples: 369
    #
    # Total data: 12190
    # Total classes: 5

    # Lấy danh sách class
    # class_names = train_dataset.classes
    # class_indices = train_dataset.class_to_idx

    # In thông tin dataset
    # print("Classes:", class_names)
    # print("Class Indices:", class_indices)
    # print(f"Train samples: {len(train_dataset)}")
    # print(f"Validation samples: {len(val_dataset)}")
    # print(f"Test samples: {len(test_dataset)}")
    # image, label = train_dataset[0]  # Lấy mẫu đầu tiên
    # print(f"Image shape: {image.shape}")  # Kích thước ảnh
    # Image shape: torch.Size([3, 224, 224])




















