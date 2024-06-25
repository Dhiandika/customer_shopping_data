# %% [markdown]
# Oleh:
#
# ```
# NIM - NAMA
# ```
#

# %% [markdown]
# About Dataset:
# Dataset ini berisi informasi tentang transaksi belanja pelanggan, dengan kolom sebagai berikut:
#
# - `invoice_no`: Nomor faktur
# - `customer_id`: ID pelanggan
# - `gender`: Jenis kelamin pelanggan
# - `age`: Usia pelanggan
# - `category`: Kategori produk
# - `quantity`: Jumlah produk yang dibeli
# - `price`: Harga total
# - `payment_method`: Metode pembayaran
# - `invoice_date`: Tanggal faktur
# - `shopping_mall`: Nama pusat perbelanjaan

# %% [markdown]
# ---
# Tugas:
# 1. Lakukan grafik untuk mengetahui keterhubungan antara age dan quantity.
# 2. lakukan grafik untuk mengetahui keterhubungan antara gender dan price
# 3. lakukan grafik untuk mengetahui jumlah payment methode (gunakan bar chart)
# 4. lakukan grafi untuk mengetahui jumlah category yang terjual (gunakan barchart)
# 5. lakukan grafik keterhubungan antara umur dengan payment method
# ---

# %% [markdown]
# # Import Libary

# %%
# Importing all the Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# %%
df = pd.read_csv('Dataset/customer_shopping_data.csv')
df.head()

# %%
df.info()

# %%
# Stastical Description of Data
df.describe()


# %%
df.shape

# %%
# Check for missing values
missing_values = df.isnull().sum()

# Check the data types
data_types = df.dtypes

missing_values, data_types

# %%
# Mengubah 'invoice_date' menjadi datetime format
df['invoice_date'] = pd.to_datetime(df['invoice_date'], dayfirst=True)

# Verify the changes
df.dtypes

# %%
df

# %% [markdown]
# ## Distribusi Dari dataset
# Distribusi data adalah cara untuk menggambarkan bagaimana data tersebar atau terseksi di sepanjang nilai-nilai yang mungkin. Ini memberikan pemahaman tentang pola atau karakteristik dari kumpulan data yang dimiliki. Pemahaman distribusi data sangat penting dalam analisis statistik karena dapat mempengaruhi pemilihan metode statistik yang tepat serta interpretasi hasil analisis.
#
#

# %%
# Create a figure with specific size
plt.figure(figsize=(18, 12))

# Define color palette
palette = 'Set3'

# Plot 1: Distribusi gender
plt.subplot(3, 3, 1)
sns.countplot(data=df, x='gender', palette=palette)
plt.title('Distribusi Gender')
plt.xlabel('Gender')
plt.ylabel('Count')

# Plot 2: Distribusi usia
plt.subplot(3, 3, 2)
sns.histplot(data=df, x='age', bins=20, kde=True, color='skyblue')
plt.title('Distribusi Usia')
plt.xlabel('Usia')
plt.ylabel('Count')

# Plot 3: Distribusi kategori produk
plt.subplot(3, 3, 3)
sns.countplot(data=df, y='category', palette=palette)
plt.title('Distribusi Kategori Produk')
plt.xlabel('Count')
plt.ylabel('Kategori Produk')

# Plot 4: Distribusi jumlah produk
plt.subplot(3, 3, 4)
sns.histplot(data=df, x='quantity', bins=20, kde=True, color='lightgreen')
plt.title('Distribusi Jumlah Produk yang Dibeli')
plt.xlabel('Jumlah Produk')
plt.ylabel('Count')

# Plot 5: Distribusi harga total
plt.subplot(3, 3, 5)
sns.histplot(data=df, x='price', bins=20, kde=True, color='salmon')
plt.title('Distribusi Total Harga')
plt.xlabel('Total Harga')
plt.ylabel('Count')

# Plot 6: Distribusi metode pembayaran
plt.subplot(3, 3, 6)
sns.countplot(data=df, x='payment_method', palette=palette)
plt.title('Distribusi Metode Pembayaran')
plt.xlabel('Metode Pembayaran')
plt.ylabel('Count')

# Plot 7: Distribusi pusat perbelanjaan (shopping mall)
plt.subplot(3, 3, 7)
sns.countplot(data=df, y='shopping_mall', palette=palette)
plt.title('Distribusi Pusat Perbelanjaan')
plt.xlabel('Count')
plt.ylabel('Pusat Perbelanjaan')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

# %% [markdown]
# ## 1. Plot grafik untuk mengetahui keterhubungan antara age dan quantity
#

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='quantity', hue='gender')
plt.title('Keterhubungan antara Age dan Quantity')
plt.xlabel('Age')
plt.ylabel('Quantity')
plt.legend(title='Gender')
plt.show()

# Define age groups
bins = [18, 25, 45, 69]
labels = ['Teen (18-25)', 'Young Adult (26-45)', 'Adult (46-69)']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

plt.figure(figsize=(10, 6))
sns.boxplot(x='age_group', y='quantity', data=df, palette='viridis')
plt.title('Relationship between Age and Quantity')
plt.xlabel('Age Group')
plt.ylabel('Quantity')
plt.show()

# %% [markdown]
# ## 2. Grafik untuk mengetahui keterhubungan antara gender dan price
#

# %%

plt.figure(figsize=(10, 6))
sns.boxplot(x='gender', y='price', data=df, palette='muted')
plt.title('Relationship between Gender and Price')
plt.xlabel('Gender')
plt.ylabel('Price')
plt.show()


# %% [markdown]
# ## 3. Grafik untuk mengetahui jumlah payment method
#

# %%
plt.figure(figsize=(10, 6))
payment_counts = df['payment_method'].value_counts()
sns.barplot(x=payment_counts.index, y=payment_counts.values)
plt.title('Jumlah Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Count')
plt.show()


# %% [markdown]
# ## 4. Grafik untuk mengetahui jumlah category yang terjual
#

# %%
plt.figure(figsize=(10, 6))
category_counts = df['category'].value_counts()
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.title('Jumlah Category yang Terjual')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()


# %% [markdown]
# ## 5. Grafik keterhubungan antara age dengan payment method
#

# %%
plt.figure(figsize=(12, 8))
sns.violinplot(data=df, x='payment_method', y='age', inner='quartile')
plt.title('Keterhubungan antara Umur dengan Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Age')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='age_group', hue='payment_method', data=df, palette='Set2')
plt.title('Relationship between Age and Payment Method')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='payment_method', y='age')
plt.title('Keterhubungan antara Usia dengan Metode Pembayaran')
plt.xlabel('Metode Pembayaran')
plt.ylabel('Usia')
plt.show()

# %% [markdown]
# ## MORE EDA

# %% [markdown]
# ### 1. Analisis Tren Waktu Transaksi
#

# %%
# Ekstrak bulan dan tahun dari kolom invoice_date
df['invoice_month'] = df['invoice_date'].dt.to_period('M')
df['invoice_year'] = df['invoice_date'].dt.year
# Hitung jumlah transaksi per bulan
transaction_per_month = df.groupby('invoice_month').size()

# %%
# Plot jumlah transaksi per bulan
plt.figure(figsize=(12, 6))
transaction_per_month.plot(kind='line', marker='o')
plt.title('Tren Jumlah Transaksi per Bulan')
plt.xlabel('Bulan')
plt.ylabel('Jumlah Transaksi')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# %% [markdown]
# ### 2. Segmentasi Pelanggan
#

# %%
# Seleksi kolom yang digunakan untuk segmentasi
X = df[['age', 'price']]

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menentukan jumlah cluster dengan metode Elbow
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Metode Elbow untuk Menentukan Jumlah Cluster')
plt.xlabel('Jumlah Cluster')
plt.ylabel('WCSS (Within Cluster Sum of Squares)')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# Berdasarkan Elbow Method, pilih jumlah cluster
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualisasi segmentasi pelanggan
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='price', hue='cluster', palette='viridis')
plt.title('Segmentasi Pelanggan berdasarkan Usia dan Total Harga')
plt.xlabel('Usia')
plt.ylabel('Total Harga')
plt.legend(title='Cluster')
plt.show()

# %% [markdown]
# ### 3.Analisis Korelasi

# %%
# Hitung korelasi antara variabel numerik
corr_matrix = df[['age', 'quantity', 'price']].corr()

# Visualisasi dengan heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
            fmt='.2f', linewidths=0.5)
plt.title('Heatmap Korelasi antara Variabel Numerik')
plt.show()

# %% [markdown]
# ### 4. Analisis outlier pada kolom price
#

# %%
# Analisis outlier pada kolom price
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='price')
plt.title('Boxplot Harga Total')
plt.xlabel('Total Harga')
plt.show()

# %% [markdown]
# ### 5. Analisis Metode Pembayaran
#

# %%
# Hitung jumlah transaksi per metode pembayaran
payment_counts = df['payment_method'].value_counts()

# Plot jumlah transaksi per metode pembayaran
plt.figure(figsize=(10, 6))
sns.barplot(x=payment_counts.index, y=payment_counts.values, palette='Set3')
plt.title('Jumlah Transaksi per Metode Pembayaran')
plt.xlabel('Metode Pembayaran')
plt.ylabel('Jumlah Transaksi')
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# ### 6. Analisis Pola Belanja Berdasarkan Kategori Produk
#

# %%
# Hitung total belanja per kategori produk
total_sales_per_category = df.groupby(
    'category')['price'].sum().sort_values(ascending=False)

# Plot total belanja per kategori produk
plt.figure(figsize=(12, 6))
total_sales_per_category.plot(kind='bar', color=sns.color_palette(
    'pastel', len(total_sales_per_category)))
plt.title('Total Belanja per Kategori Produk')
plt.xlabel('Kategori Produk')
plt.ylabel('Total Belanja')
plt.xticks(rotation=45)
plt.show()


# %% [markdown]
# ### 7. Analisis Tren Harian Transaksi
#

# %%
# Ekstrak hari dari kolom invoice_date
df['invoice_day'] = df['invoice_date'].dt.day_name()

# Hitung jumlah transaksi per hari
transaction_per_day = df['invoice_day'].value_counts()

# Urutkan berdasarkan urutan hari dalam seminggu
transaction_per_day = transaction_per_day.reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Plot jumlah transaksi per hari
plt.figure(figsize=(10, 6))
sns.barplot(x=transaction_per_day.index,
            y=transaction_per_day.values, palette='rainbow')
plt.title('Jumlah Transaksi per Hari')
plt.xlabel('Hari')
plt.ylabel('Jumlah Transaksi')
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# ### 8. Segmentasi Pelanggan Berdasarkan Gender dan Metode Pembayaran
#

# %%
# Hitung jumlah transaksi per kombinasi gender dan metode pembayaran
gender_payment_counts = df.groupby(
    ['gender', 'payment_method']).size().reset_index(name='count')

# Plot segmentasi pelanggan
plt.figure(figsize=(10, 6))
sns.barplot(data=gender_payment_counts, x='gender',
            y='count', hue='payment_method', palette='Set2')
plt.title('Segmentasi Pelanggan berdasarkan Gender dan Metode Pembayaran')
plt.xlabel('Gender')
plt.ylabel('Jumlah Transaksi')
plt.legend(title='Metode Pembayaran')
plt.show()

# %%
