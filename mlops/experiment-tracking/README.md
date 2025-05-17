# Experiment Tracking
Ada beberapa hal yang bisa di-highlight di bagian Experiment Tracking
- Gunakan autolog pertama kali running experiment untuk bisa mengetahui parameter apa saja yang bisa digunakan.
- Setiap framework memiliki keunggulan masing-masing yang bisa diintegrasikan dengan mlflow. Dalam kasus ini, tensorflow digunakan untuk log model.
- Kita bisa menambahkan log data input dari data explorations sampai model evaluation. Kalau datanya tabular atau pandas bisa menggunakan `mlflow.data.dataset.from_pandas` (keyword: MLFlow dataset tracking). Kasus ini pakai CV menggunakan image, jadi belum nemu trackingnya pakai apa. 
- Proses tracking bisa beragam bergantung pada informasi yang ingin dicari, tapi secara umum yang dicari berupa **Hyperparameter,Metrics Evaluasi, Dataset, Informasi Running (seperti author dan time)**
- MLFlow memiliki model_signature untuk tracking input dan output model nanti ketika inference agar terstandar.


# Catatan
Berikut adalah hasil training dari model CNN yang dilatih.
![Hasil Training](./image-docs/training-xray.png)