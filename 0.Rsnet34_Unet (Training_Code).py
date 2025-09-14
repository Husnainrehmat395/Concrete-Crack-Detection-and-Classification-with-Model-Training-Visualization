{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "  \n",
        "---\n",
        "\n",
        "# **LOADING DATA FROM ROBOFLOW**\n",
        "\n",
        "---\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "nEiGrL2qGkyg"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MXMBO5s6-dFT",
        "outputId": "858c53ba-acec-4e86-e524-e7b78580d530"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: roboflow in /usr/local/lib/python3.11/dist-packages (1.1.61)\n",
            "Requirement already satisfied: certifi in /usr/local/lib/python3.11/dist-packages (from roboflow) (2025.1.31)\n",
            "Requirement already satisfied: idna==3.7 in /usr/local/lib/python3.11/dist-packages (from roboflow) (3.7)\n",
            "Requirement already satisfied: cycler in /usr/local/lib/python3.11/dist-packages (from roboflow) (0.12.1)\n",
            "Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.11/dist-packages (from roboflow) (1.4.8)\n",
            "Requirement already satisfied: matplotlib in /usr/local/lib/python3.11/dist-packages (from roboflow) (3.10.0)\n",
            "Requirement already satisfied: numpy>=1.18.5 in /usr/local/lib/python3.11/dist-packages (from roboflow) (2.0.2)\n",
            "Requirement already satisfied: opencv-python-headless==4.10.0.84 in /usr/local/lib/python3.11/dist-packages (from roboflow) (4.10.0.84)\n",
            "Requirement already satisfied: Pillow>=7.1.2 in /usr/local/lib/python3.11/dist-packages (from roboflow) (11.1.0)\n",
            "Requirement already satisfied: pillow-heif>=0.18.0 in /usr/local/lib/python3.11/dist-packages (from roboflow) (0.22.0)\n",
            "Requirement already satisfied: python-dateutil in /usr/local/lib/python3.11/dist-packages (from roboflow) (2.8.2)\n",
            "Requirement already satisfied: python-dotenv in /usr/local/lib/python3.11/dist-packages (from roboflow) (1.1.0)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from roboflow) (2.32.3)\n",
            "Requirement already satisfied: six in /usr/local/lib/python3.11/dist-packages (from roboflow) (1.17.0)\n",
            "Requirement already satisfied: urllib3>=1.26.6 in /usr/local/lib/python3.11/dist-packages (from roboflow) (2.3.0)\n",
            "Requirement already satisfied: tqdm>=4.41.0 in /usr/local/lib/python3.11/dist-packages (from roboflow) (4.67.1)\n",
            "Requirement already satisfied: PyYAML>=5.3.1 in /usr/local/lib/python3.11/dist-packages (from roboflow) (6.0.2)\n",
            "Requirement already satisfied: requests-toolbelt in /usr/local/lib/python3.11/dist-packages (from roboflow) (1.0.0)\n",
            "Requirement already satisfied: filetype in /usr/local/lib/python3.11/dist-packages (from roboflow) (1.2.0)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib->roboflow) (1.3.2)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib->roboflow) (4.57.0)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib->roboflow) (24.2)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib->roboflow) (3.2.3)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->roboflow) (3.4.1)\n",
            "loading Roboflow workspace...\n",
            "loading Roboflow project...\n"
          ]
        }
      ],
      "source": [
        "!pip install roboflow\n",
        "\n",
        "from roboflow import Roboflow\n",
        "rf = Roboflow(api_key=\"8tK1IESi8zUOKiF3UbOh\")\n",
        "project = rf.workspace(\"acd2\").project(\"acd_sg_19\")\n",
        "version = project.version(5)\n",
        "dataset = version.download(\"coco-segmentation\")"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "---\n",
        "\n",
        "# **Install TensorFlow & Mount Google Drive**\n",
        "\n",
        "---\n",
        "\n"
      ],
      "metadata": {
        "id": "S3m9u2KDG47C"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install tensorflow\n",
        "import tensorflow as tf\n",
        "import os\n",
        "import json\n",
        "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
        "# Mount the Google Drive at /content/drive\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fajQVisb_Y7P",
        "outputId": "b7bcdaeb-08a5-46d7-d9bf-3cbcce472c22"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: tensorflow in /usr/local/lib/python3.11/dist-packages (2.18.0)\n",
            "Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.4.0)\n",
            "Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.6.3)\n",
            "Requirement already satisfied: flatbuffers>=24.3.25 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (25.2.10)\n",
            "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (0.6.0)\n",
            "Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (0.2.0)\n",
            "Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (18.1.1)\n",
            "Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (3.4.0)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from tensorflow) (24.2)\n",
            "Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<6.0.0dev,>=3.20.3 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (5.29.4)\n",
            "Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (2.32.3)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.11/dist-packages (from tensorflow) (75.2.0)\n",
            "Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.17.0)\n",
            "Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (3.0.1)\n",
            "Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (4.13.2)\n",
            "Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.17.2)\n",
            "Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.71.0)\n",
            "Requirement already satisfied: tensorboard<2.19,>=2.18 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (2.18.0)\n",
            "Requirement already satisfied: keras>=3.5.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (3.8.0)\n",
            "Requirement already satisfied: numpy<2.1.0,>=1.26.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (2.0.2)\n",
            "Requirement already satisfied: h5py>=3.11.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (3.13.0)\n",
            "Requirement already satisfied: ml-dtypes<0.5.0,>=0.4.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (0.4.1)\n",
            "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (0.37.1)\n",
            "Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.11/dist-packages (from astunparse>=1.6.0->tensorflow) (0.45.1)\n",
            "Requirement already satisfied: rich in /usr/local/lib/python3.11/dist-packages (from keras>=3.5.0->tensorflow) (13.9.4)\n",
            "Requirement already satisfied: namex in /usr/local/lib/python3.11/dist-packages (from keras>=3.5.0->tensorflow) (0.0.8)\n",
            "Requirement already satisfied: optree in /usr/local/lib/python3.11/dist-packages (from keras>=3.5.0->tensorflow) (0.15.0)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.4.1)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.7)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.21.0->tensorflow) (2.3.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.21.0->tensorflow) (2025.1.31)\n",
            "Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.11/dist-packages (from tensorboard<2.19,>=2.18->tensorflow) (3.8)\n",
            "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.11/dist-packages (from tensorboard<2.19,>=2.18->tensorflow) (0.7.2)\n",
            "Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from tensorboard<2.19,>=2.18->tensorflow) (3.1.3)\n",
            "Requirement already satisfied: MarkupSafe>=2.1.1 in /usr/local/lib/python3.11/dist-packages (from werkzeug>=1.0.1->tensorboard<2.19,>=2.18->tensorflow) (3.0.2)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.11/dist-packages (from rich->keras>=3.5.0->tensorflow) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.11/dist-packages (from rich->keras>=3.5.0->tensorflow) (2.18.0)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.11/dist-packages (from markdown-it-py>=2.2.0->rich->keras>=3.5.0->tensorflow) (0.1.2)\n",
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "---\n",
        "\n",
        "# **Set Dataset Paths & Load COCO Annotations**\n",
        "\n",
        "---\n"
      ],
      "metadata": {
        "id": "9z_av8XnIO87"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "dataset_path = \"/content/drive/MyDrive/CrackDetection\"\n",
        "train_path = f\"{dataset_path}/train\"\n",
        "val_path = f\"{dataset_path}/valid\"\n",
        "test_path = f\"{dataset_path}/test\"\n",
        "\n",
        "annotations_file = '/content/drive/My Drive/CrackDetection/train/_annotations.coco.json'\n",
        "\n",
        "with open(annotations_file) as f:\n",
        "    annotations = json.load(f)\n",
        "\n",
        "# Checking the structure of the annotations\n",
        "print(annotations.keys())\n",
        "print(f\"Number of annotations: {len(annotations['annotations'])}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OWXVOYMj_rcK",
        "outputId": "ccc7e47e-5c55-409c-a5d3-c838b0c4bee6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "dict_keys(['info', 'licenses', 'categories', 'images', 'annotations'])\n",
            "Number of annotations: 6567\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "---\n",
        "\n",
        "# **Map Images to Their Crack Type Annotations**\n",
        "\n",
        "---\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "_YAB-JAEI4AV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "image_annotations = {}\n",
        "\n",
        "for annotation in annotations['annotations']:\n",
        "    image_id = annotation['image_id']\n",
        "    crack_type = annotation['category_id']\n",
        "    image_file = annotations['images'][image_id - 1]['file_name']\n",
        "\n",
        "    if image_file not in image_annotations:\n",
        "        image_annotations[image_file] = []\n",
        "    image_annotations[image_file].append(crack_type)\n",
        "\n",
        "print(f\"First image annotations: {image_annotations}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gFa-zVJU_tJ_",
        "outputId": "d40aab97-0703-4c0d-83ba-2659a709085f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "First image annotations: {'C-1300-5_560_2msX-134-_jpg.rf.80efa5b54da90ad6313fe4e8301c7f4c.jpg': [4, 2, 2, 2, 2, 2, 1, 5], 'B-1052-6_40-0_2msX-29-_jpg.rf.caf291de963624f31d67e620a76ad527.jpg': [4, 4, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'C-868-_7140_2msX-107-_jpg.rf.4354ad8ee7abc5609d2376fe33f378f6.jpg': [4, 1], 'cropped_868kg0_2msX-439-_jpg.rf.af6023d1393ad67644973b1e0672c9b5.jpg': [5, 4, 4, 4, 4, 4, 1], 'D-868-7_140_2msX-49-_jpg.rf.f33de882615b40e862090f111f9f33f0.jpg': [4, 2, 2, 2, 2, 2, 2, 2, 5, 1], 'B-1052-6_40-0_2msX-31-_jpg.rf.070377d8de07f15850082ee14bb26c5c.jpg': [4, 4, 5, 5, 2, 4, 1, 1, 3], 'cropped_B-1300-5_56-0_2mmX-180-_jpg.rf.a73a5c77350a36d65a60874b3c499190.jpg': [3, 1, 1, 2, 4, 1, 4], 'cropped_C-1300-5_56-0_2X-171-_png.rf.8bbba4e58e9f2dc894ac309a92f4a7bf.jpg': [3, 3, 4, 2, 4, 4, 5, 5, 5, 5, 3, 1], '-0_2X-375-_jpg.rf.90a8ac951f8fb70eb452c1366b5041c4.jpg': [3, 2, 2, 2, 4, 1, 1, 1, 1], 'cropped_C-1300-5_56-0_2X-114-_png.rf.2dd1cc55f540fa2031043a2e61606fb2.jpg': [3, 5, 3, 4, 4, 5, 5, 5, 1, 1, 2], 'cropped_B-1700-4_60-0_2X-287-_jpg.rf.4b594265bb72949c82065dbcfd23659e.jpg': [4, 2, 1, 1, 1, 1, 1], 'C-868-_7140_2msX-68-_jpg.rf.cf96c00d920efbdb7595468d1240142a.jpg': [1, 1, 1, 5, 4, 4, 5, 4, 5, 5, 1, 1, 1], 'D-868-7_140_2msX-69-_jpg.rf.e465778cdceda9577a9ce8927d18b5d7.jpg': [4, 1], 'cropped_868kg0_2msX-157-_jpg.rf.5da4f4493dccc9207b12c1e1e21dc8fc.jpg': [2, 2, 2, 1, 1, 4, 1, 1, 1, 3, 3, 3, 3, 5, 2, 5], 'C-1300-5_560_2msX-73-_jpg.rf.2e7c6ca665acdf86b3633289f37c5132.jpg': [4, 4, 4, 1, 5, 4, 3], 'cropped_B-1300-5_56-0_2mmX-81-_jpg.rf.95c7843082e1eec6fdbec87b99cf0fe4.jpg': [4, 2, 2, 2, 2, 2, 1, 5], 'B-1052-6_40-0_2msX-29-_jpg.rf.7cfa0e50d69920b2a5b2c9e8e9846ecb.jpg': [3, 5, 3, 4, 4, 5, 5, 5, 1, 1, 2], 'cropped_B-1700-4_60-0_2X-287-_jpg.rf.ada9494bd87f2edc71b03c99ad2a0425.jpg': [3, 3, 4, 2, 4, 4, 5, 5, 5, 5, 3, 1], '-0_2X-375-_jpg.rf.06014959b13088898544dc495e74c4aa.jpg': [4, 4, 4, 5, 5, 3, 2, 4, 4, 1, 1, 1], 'C-1300-5_560_2msX-27-_jpg.rf.f2a3835f036c257066d7580e555071a5.jpg': [4, 4, 4, 5, 5, 3, 2, 4, 4, 1, 1, 1], 'C-1300-5_560_2msX-27-_jpg.rf.3f4f0efa6c88e2b2c4b05f643c4829ec.jpg': [4, 5, 1], 'B-1052-6_40-0_2msX-8-_jpg.rf.743ff8473c86c2591cb983479721e74e.jpg': [1, 1, 1, 5, 5, 4, 4, 4], 'cropped_D-868-7_140_2msX-133-_jpg.rf.380ee92e932e5e3e9b80a273ccc45065.jpg': [4, 2, 2, 2, 1], 'cropped_B-1052-6_40-0_2msX-27-_jpg.rf.f0bc23ff2bc87cf88272bdaf197a9457.jpg': [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 3, 4, 5, 5, 1, 3, 1, 3, 5, 2, 3, 1], '-0_2X-18-_jpg.rf.6734f8d7f7c58503a3b04fc5c690db24.jpg': [1, 5, 4, 4, 4, 4], 'cropped_D-868-7_140_2msX-79-_jpg.rf.1a3aa31e2cf5470bc9d34f080ebfb5b1.jpg': [1, 4, 4, 4, 4], 'cropped_D-868-7_140_2msX-23-_jpg.rf.e12293ebd152e0d5b3be03aac3bbaff0.jpg': [3, 2, 1, 1, 1, 4, 1, 3, 5, 5], 'C-1300-5_560_2msX-134-_jpg.rf.1b6951db0b82f6e23c520eb97dbaccaa.jpg': [4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1], 'C-868-_7140_2msX-64-_jpg.rf.dce4734a3d861cba0c11e5fa39fd5ca2.jpg': [5, 4, 1, 1, 5, 5, 2], '868kg0_2msX-130-_jpg.rf.7d12ed1865d451370ee0406720bcf2fb.jpg': [3, 1, 4], 'cropped_C-1700-4_600_2mmX-51-_jpg.rf.11ce7e058d8b87b8b1e0934d517ee21b.jpg': [3, 1, 1, 1], 'cropped_C-1700-4_600_2mmX-197-_jpg.rf.d38e55d0b8e3262da51664870812b8be.jpg': [3, 3, 3, 2, 1, 1, 1, 4, 4, 3, 1, 1], 'C-1300-5_560_2msX-104-_jpg.rf.5ebc2b59e5616f075f913e3dd5c118b9.jpg': [2, 4, 5, 2, 2, 2, 3, 5, 1, 1, 1, 1, 1], 'B-1052-6_40-0_2msX-174-_jpg.rf.deeb9ebbd32c36581d2fa15e2c2e97d3.jpg': [1, 1, 1, 5, 5, 4, 4, 4], 'cropped_D-868-7_140_2msX-133-_jpg.rf.432407de8ea971de878e1659be12fb7f.jpg': [4, 4, 5, 5, 1, 4, 3, 2, 2], 'cropped_B-1300-5_56-0_2mmX-147-_jpg.rf.c341abd0e71e05fbdb025fb1fd079301.jpg': [4, 4, 4, 4, 2, 2, 2, 3, 3, 3, 4, 5, 1, 2], 'B1330-5_56-0_2msX-91-_jpg.rf.b704cb5b5ff802d2811de24c0a362c1c.jpg': [4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'C-868-_7140_2msX-147-_jpg.rf.2a3346898ca569a65990fd65672ced5e.jpg': [5, 5, 4, 2, 3, 1, 1, 1, 1, 1, 3], 'cropped_B-1700-4_60-0_2X-327-_jpg.rf.668dbab329bfefaf0aba1284a6bbfd04.jpg': [2, 1, 4, 5, 5, 3, 4, 1, 3], 'C-1300-5_560_2msX-113-_jpg.rf.f15b388fb483a640f3c95bea29799603.jpg': [3, 3, 4, 4, 4, 5, 5, 3, 3, 2, 2, 1, 1, 1, 1], 'C-1300-5_560_2msX-41-_jpg.rf.c0b8a40b69e9e56f2d95521fe176d2a0.jpg': [3, 3, 1, 1, 5, 4, 1], 'C-1700-4_600_2mmX-195-_jpg.rf.e5584a62584d68351538a061b7f02017.jpg': [4, 1], 'cropped_C-868-_7140_2msX-47-_jpg.rf.f08ba812db5f7c55f4a13860d5b94588.jpg': [2, 4, 2, 5, 3], 'cropped_B-1052-6_40-0_2msX-88-_jpg.rf.ee80df510c1a5dab6d5f1d2db570070b.jpg': [4, 1], 'cropped_868kg0_2msX-344-_jpg.rf.e43e9b94dd3f29b998c62727957fbe22.jpg': [5, 4, 1, 3, 3, 1, 1, 3, 3, 2], 'C-1300-5_560_2msX-172-_jpg.rf.cff9c6780395edd897411f8f3c528947.jpg': [3, 1], 'cropped_C-1700-4_600_2mmX-115-_jpg.rf.c868ea0658515859b12c6922a5dc238b.jpg': [3, 4, 4, 2, 2, 1, 1, 3, 5, 5, 5, 5], 'cropped_B-1700-4_60-0_2X-186-_jpg.rf.149dcdfe622d20ea4bd50a95b93fe947.jpg': [4, 4, 4, 1, 5, 5, 4, 4], 'D-1700-4_600_5msX-43-_jpg.rf.f88969aba4f60c71b3d7a582b650a900.jpg': [4, 4, 4, 4, 4, 2, 2, 2, 3, 3, 3], 'B1330-5_56-0_2msX-56-_jpg.rf.3539fecb7ffc67e4695ab65e52c8910c.jpg': [1, 5, 4, 1, 3], 'C-1700-4_600_2mmX-97-_jpg.rf.5efc496c2d1955e956b45f58a848940e.jpg': [4, 3, 1, 5, 5, 5, 4], 'D-1700-4_600_5msX-49-_jpg.rf.9393ca0543203f6cad306d69e0a1719c.jpg': [1, 5, 5, 4, 4, 4, 1, 1], 'cropped_D-868-7_140_2msX-200-_jpg.rf.779f2b8c1735ba1cc3fa4fb2244ea64d.jpg': [4, 4, 4, 3], 'cropped_B-1300-5_56-0_2mmX-49-_jpg.rf.0ba45caa1304636f2e30d8f195c86cd9.jpg': [4, 1], 'cropped_C-868-_7140_2msX-70-_jpg.rf.8e66df605d11e04161f342d684efa8cc.jpg': [2, 2, 4, 1, 5, 3], 'cropped_B-1052-6_40-0_2msX-100-_jpg.rf.042481ede142c7ef450f89fb9f68cf5a.jpg': [3, 4, 1], 'cropped_C-1700-4_600_2mmX-24-_jpg.rf.87c698fcc472eb66194aaba674b3d1f1.jpg': [3, 4, 2, 2, 4, 4, 4, 5, 5, 3, 5, 5, 1, 1], '-0_2X-211-_jpg.rf.7c8bda0c4e76273f7e9109e457747c7a.jpg': [3, 3, 3, 4], 'D-1700-4_600_5msX-8-_jpg.rf.a484ff8ffb6e55b6aa6e88a5b5628dd2.jpg': [3, 1, 2, 5, 2, 4, 4, 3, 1, 1, 1, 1], 'C-1300-5_560_2msX-103-_jpg.rf.426da82fadd2b0f2520bb0c5fc57d59c.jpg': [1, 4, 2, 4, 3, 3, 3, 5, 4, 1, 1, 1, 5, 5], 'C-1300-5_560_2msX-105-_jpg.rf.7d70dc22c1e6adf2c748df313a105513.jpg': [5, 4, 4, 2, 1, 5], '868kg0_2msX-480-_jpg.rf.6c20156ee755d6d9b334954de6128551.jpg': [1, 5, 1, 4, 5, 4], '868kg0_2msX-18-_jpg.rf.a34ffc213fd358f9bd1b932c64f371f6.jpg': [4, 2, 2, 2, 2, 1, 5], 'B-1052-6_40-0_2msX-12-_jpg.rf.48be6f12d4456957b7d8da069c3564f5.jpg': [4, 1, 1], 'cropped_868kg0_2msX-190-_jpg.rf.ddb6c8b1673bd04cfb8b61f2ea14b1b4.jpg': [4, 2, 2, 2, 2, 2, 5, 1, 5], 'B-1052-6_40-0_2msX-27-_jpg.rf.aff2acd9133a2b99bc6a9cce18b95b0d.jpg': [1, 1, 4, 4, 4, 5], 'D-868-7_140_2msX-23-_jpg.rf.9c9c126fea02778a65f3bfc6e75d84fa.jpg': [3, 4, 2, 2, 4, 4, 5, 5, 3, 1], '-0_2X-165-_jpg.rf.04e91cbc86dd2e409e44645bc78b6fcc.jpg': [5, 5, 4, 5, 4, 3, 3, 1], 'D-1700-4_600_5msX-48-_jpg.rf.69de9c83b3ad9e1f64578c8daea769dc.jpg': [3, 4, 2, 2, 4, 4, 5, 5, 5, 5, 3, 1], '-0_2X-200-_jpg.rf.53b5c4c8b72195fb2064ca0b883e2c8f.jpg': [4, 5, 1], 'cropped_868kg0_2msX-25-_jpg.rf.c77add33f24f041d74f78de82b470f07.jpg': [1, 1, 1, 2, 3, 3, 5, 2, 2, 2, 2, 4, 2, 4, 1, 3, 1, 1], 'C-1300-5_560_2msX-70-_jpg.rf.f9aa8d28cd9ccb75a3d698318876caee.jpg': [3, 1, 1, 1, 5, 5, 4], 'C-1700-4_600_2mmX-171-_jpg.rf.ec6a54f84953268de773606aba0a8b61.jpg': [4, 4, 5, 3], 'cropped_B-1300-5_56-0_2mmX-25-_jpg.rf.2b7ba3e70c49554f52454a4cdbf87521.jpg': [4, 4, 4, 4], 'C-1700-4_600_2mmX-18-_jpg.rf.cca415bb3a1156731b94878886a614eb.jpg': [1, 4, 5, 5, 2, 2, 2, 2, 3, 3, 5, 5, 1], 'B-1052-6_40-0_2msX-85-_jpg.rf.e26977811387862d7a62a7f23c743e0d.jpg': [4, 2, 2, 4, 1], 'cropped_C-868-_7140_2msX-106-_jpg.rf.43aef2fc568591b17e9159e2e81a0d90.jpg': [4, 4, 4, 4], 'C-1700-4_600_2mmX-24-_jpg.rf.c291e9e72228954593546655dfd2cb77.jpg': [2, 5, 3, 3, 5, 2, 2, 2, 4, 4, 4, 4, 5, 5, 1, 1, 1, 3, 1, 1, 2, 4], 'C-1300-5_560_2msX-46-_jpg.rf.693ce5b8b5a194a1453643b93d1d13d2.jpg': [4, 4, 4, 5, 5], 'cropped_B-1300-5_56-0_2mmX-13-_jpg.rf.eddfc5507ba6a3a44a63b77988f3f1f9.jpg': [4, 4, 4, 4, 1, 1], 'D-868-7_140_2msX-22-_jpg.rf.094e68666c201e860cae66521f21df4e.jpg': [2, 1, 4, 5, 5, 3, 4, 1, 3], 'C-1300-5_560_2msX-113-_jpg.rf.d8422051f0c060181aa6a43f1809f843.jpg': [3, 4, 5, 2, 4, 2, 3, 1], 'cropped_B-1700-4_60-0_2X-87-_jpg.rf.92a5c564958f1733c77a220166636a84.jpg': [3, 1, 1, 1, 2, 4], 'cropped_C-1300-5_56-0_2X-96-_png.rf.7b93ea84c7518c0c9f8908f8ca39f9b1.jpg': [5, 4, 1, 3, 3, 1, 1, 3, 3, 2], 'C-1300-5_560_2msX-172-_jpg.rf.66b486882c7154b9cedadd378f616739.jpg': [5, 4, 4, 4, 4, 1, 1, 1, 1, 1, 5, 5, 4, 4], 'D-868-7_140_2msX-68-_jpg.rf.99c56a96e632911756560e8375412174.jpg': [4, 4, 5, 5, 1, 4, 3, 2, 2], 'cropped_B-1300-5_56-0_2mmX-147-_jpg.rf.9e3aa8d2e6fafc23e7fd97d2d8690f35.jpg': [4, 2, 2, 1, 1, 1, 1], 'C-868-_7140_2msX-66-_jpg.rf.10e559cb94289c93ffdd6c07923e2bf5.jpg': [4, 4, 4, 4, 2, 2, 3, 3, 3, 2, 2], 'B1330-5_56-0_2msX-53-_jpg.rf.dd2e25a803a51102d99c24f522cad0a9.jpg': [3, 2, 2, 1, 1, 5, 3, 2, 2, 2, 4, 4, 1, 1, 1, 1, 1], 'C-1300-5_560_2msX-67-_jpg.rf.28248eecf2f3df7c6e49f8a43ae7c5b1.jpg': [3, 3, 1, 3, 4, 1, 1], 'C-1700-4_600_2mmX-44-_jpg.rf.d41b2a4fd882c5858fb47851e6e28d07.jpg': [1, 2, 4, 5, 5], '868kg0_2msX-404-_jpg.rf.1caa574ba50be058e2a924f02d21353a.jpg': [4, 2, 2, 2, 4, 4], '-0_2X-19-_jpg.rf.800f0a5a7391fff6d4ebd0af32035c3d.jpg': [3, 4, 4, 4, 2, 4, 4, 5, 5, 5, 1, 1, 4, 3, 5], 'C-1300-5_560_2msX-23-_jpg.rf.fb1b28c92badafba2bda5cd3897b33c1.jpg': [3, 4, 2, 2, 4, 5, 4, 3, 1, 1, 1], '-0_2X-64-_jpg.rf.72ba36259ba8b3eb51bc4515c1e8a5e2.jpg': [1, 4, 5, 5, 2, 2, 2, 2, 3, 3, 3, 5, 1], 'B-1052-6_40-0_2msX-83-_jpg.rf.8f119488d0ac5f30d7aec7d6f5a76737.jpg': [5, 4, 5, 4, 1, 4, 4], 'cropped_B-1300-5_56-0_2mmX-194-_jpg.rf.77a675bcb697458584c9a95fb23b46e6.jpg': [5, 4, 4, 4, 1, 5], '868kg0_2msX-8-_jpg.rf.7a769eb028c75a34eb9ded247c802570.jpg': [4, 4, 2, 3, 1, 1, 5, 5], 'cropped_B-1300-5_56-0_2mmX-113-_jpg.rf.119f132b363d6d62e5ab7cb4b73db84c.jpg': [1, 4, 5, 5, 5, 2, 2, 2, 2, 3, 3, 2, 1], 'B-1052-6_40-0_2msX-82-_jpg.rf.9f03655fb46065110b63d5047b6bdfcc.jpg': [1, 4, 2, 4, 3, 3, 3, 5, 4, 1, 1, 1, 5, 5], 'C-1300-5_560_2msX-105-_jpg.rf.c344e53028272f2065b741b6fed48e13.jpg': [5, 4, 4, 4, 4, 4, 1, 1, 1, 5, 4], 'D-868-7_140_2msX-106-_jpg.rf.50cb8b3a61af6f734c8aa100c6e4276c.jpg': [1, 5, 4, 1, 3], 'C-1700-4_600_2mmX-97-_jpg.rf.185754c90f0feb3362f6248f93a54649.jpg': [5, 5, 4, 4, 3, 3, 1, 4, 5], 'D-1700-4_600_5msX-13-_jpg.rf.3e532aff6d0eb1877f14634bad1cfcde.jpg': [1, 5, 1, 4, 5, 4], '868kg0_2msX-18-_jpg.rf.b40d6a5e7e476a0af2305ef4ee4e10ec.jpg': [5, 5, 4, 5, 4, 4, 4, 1], 'D-868-7_140_2msX-35-_jpg.rf.707dc8523fcae25481981297116a7843.jpg': [2, 2, 4, 1, 5, 3], 'cropped_B-1052-6_40-0_2msX-100-_jpg.rf.c939ff3f17deb3d3051f03f72c1e5c2f.jpg': [4, 3, 1, 5, 5, 5, 4], 'D-1700-4_600_5msX-49-_jpg.rf.79450152eb52e5d36b9b18e93d061147.jpg': [4, 4, 4, 4, 2, 2, 2, 3, 3, 3, 4, 5, 1, 2], 'B1330-5_56-0_2msX-91-_jpg.rf.b802aa14259df9bbda843837a264eefb.jpg': [1, 1, 5, 4, 2, 2, 2, 2, 3], 'cropped_B-1052-6_40-0_2msX-159-_jpg.rf.96da49f875df498bf4da74f524309af3.jpg': [4, 1], 'cropped_C-868-_7140_2msX-70-_jpg.rf.521582388aea709650939dacf1c3b05a.jpg': [1, 1, 1, 2, 2, 2, 4, 3, 3, 5, 5, 5, 1, 1], 'C-1300-5_560_2msX-75-_jpg.rf.0e25a9b0b3eaa58337a5ad98eb7a99b5.jpg': [1, 4, 5, 5, 2, 2, 2, 2, 3, 3, 5, 5, 1], 'B-1052-6_40-0_2msX-85-_jpg.rf.3492323ec718344520ce3b4675b9559d.jpg': [5, 5, 4, 2, 3, 1, 1, 1, 1, 1, 3], 'cropped_B-1700-4_60-0_2X-327-_jpg.rf.47da4d1daddae1a985d9d4b5bc5b2a94.jpg': [4, 4, 4, 4, 2, 2, 4, 3, 3, 3], 'B1330-5_56-0_2msX-90-_jpg.rf.6da9feb2617d444bdd00f65a7009730a.jpg': [4, 3, 3, 1, 5, 5, 5, 4], 'D-1700-4_600_5msX-46-_jpg.rf.d0c24218d59dd089cb9957296b594c52.jpg': [4, 4, 4, 1, 5, 5, 4, 4], 'D-1700-4_600_5msX-43-_jpg.rf.773966ccd13622b9461761270b01e4da.jpg': [3, 1, 1, 1, 5, 5, 4], 'C-1700-4_600_2mmX-171-_jpg.rf.08b757cb4d36dfabfbb9cd4f907dc3ca.jpg': [3, 1, 1, 1, 2, 4], 'cropped_C-1300-5_56-0_2X-96-_png.rf.b987bf9c401657d50072ce86589d6f97.jpg': [3, 4, 1], 'cropped_C-1700-4_600_2mmX-24-_jpg.rf.5fd8e25fc3ec52b881e59ef65d0bfc71.jpg': [3, 4, 2, 4, 4, 5, 5, 5, 5, 3, 1, 1, 1], '-0_2X-372-_jpg.rf.2c3645367ec6822f3b4c023fa58ccacd.jpg': [4, 4, 4, 3], 'cropped_B-1300-5_56-0_2mmX-49-_jpg.rf.55f0cd2f33434ad3a26960319d9d799d.jpg': [5, 4, 4, 2, 1, 5], '868kg0_2msX-480-_jpg.rf.5848b6e5e507f900312d00ee08178dd8.jpg': [4], 'cropped_868kg0_2msX-457-_jpg.rf.6210a27f1d10f475f32618eeb8a7c706.jpg': [2, 2, 4, 5, 3], 'cropped_B-1052-6_40-0_2msX-116-_jpg.rf.66b1429f97da254fdf21386c860d10a0.jpg': [4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'C-868-_7140_2msX-147-_jpg.rf.43caead8aa47d12e76539983fc796cdd.jpg': [3, 1, 2, 5, 2, 4, 4, 3, 1, 1, 1, 1], 'C-1300-5_560_2msX-103-_jpg.rf.d3171dfeaf59e0d26bd7cc2af59ff2c4.jpg': [4, 4, 4, 5], 'cropped_B-1300-5_56-0_2mmX-13-_jpg.rf.9d85a1bbea4723d8cdd73db277b26abf.jpg': [3, 3, 4, 4, 4, 5, 5, 3, 3, 2, 2, 1, 1, 1, 1], 'C-1300-5_560_2msX-41-_jpg.rf.f29d011e507e5183a4ef42764531fc67.jpg': [1, 1, 5, 4, 2, 2, 2, 2, 3], 'cropped_B-1052-6_40-0_2msX-159-_jpg.rf.2e3a294e0f861050f0b464864fcada23.jpg': [4, 4, 2, 3, 1, 1, 5, 5], 'cropped_B-1300-5_56-0_2mmX-113-_jpg.rf.de9eee10ce877f999e80b4f9b047826a.jpg': [2, 2, 4, 5, 3], 'cropped_B-1052-6_40-0_2msX-116-_jpg.rf.84104cc774d7e3973d2498d9fcb2c271.jpg': [3, 1], 'cropped_C-1700-4_600_2mmX-115-_jpg.rf.41b15dc1c307238ac14d20feed354549.jpg': [4, 4, 5, 4, 1, 3], 'cropped_B-1300-5_56-0_2mmX-125-_jpg.rf.1a01ef8f35d69493b1a0d93dfc04ac17.jpg': [1, 1, 1, 2, 2, 2, 4, 3, 3, 5, 5, 5, 1, 1], 'C-1300-5_560_2msX-75-_jpg.rf.588cbe2a41357528852f27f63e201876.jpg': [3, 1, 4], 'cropped_C-1300-5_56-0_2X-22-_png.rf.067b673f056f4c75e26b15a35c0f3ab1.jpg': [4, 5, 5, 4, 5, 4, 1, 3, 4], 'cropped_B-1300-5_56-0_2mmX-136-_jpg.rf.4af8dd470e98a93661139632f19a85bd.jpg': [4, 5, 5, 4, 5, 4, 1, 3, 4], 'cropped_B-1300-5_56-0_2mmX-136-_jpg.rf.8105967747a0a3ff23cfafed1d38752d.jpg': [1, 1, 5, 4, 2, 2, 2, 2, 3], 'cropped_B-1052-6_40-0_2msX-159-_jpg.rf.31c89241bcb16bc9a21489cdaa6fa5b7.jpg': [3, 5, 2, 2, 4, 4, 4, 4, 5, 5, 5, 3, 1, 1, 1], '-0_2X-201-_jpg.rf.322de1a1e4f163fd1dbb6e11dd97c2c6.jpg': [4, 2, 2, 2, 4, 4], '-0_2X-19-_jpg.rf.b435a1b28af97f1cc7117b389dbf54e4.jpg': [5, 4, 4, 4, 4, 4, 1, 1, 1, 5, 4], 'D-868-7_140_2msX-106-_jpg.rf.8eed52f7da5ae66ddc93a340fccff1c3.jpg': [3, 4, 2, 2, 4, 4, 4, 5, 3], '-0_2X-65-_jpg.rf.1366826e172801bfbbcdbcf0ff1860f2.jpg': [4, 1, 1, 2, 2], 'C-868-_7140_2msX-34-_jpg.rf.ac0d83f12a6c1a223306e2ad5a99775b.jpg': [3, 3, 1, 3, 4, 1, 1], 'C-1700-4_600_2mmX-44-_jpg.rf.4f0f76dc8be7a1b9ed3a99801432762d.jpg': [1, 1, 1, 2, 2, 2, 4, 3, 3, 5, 5, 5, 1, 1], 'C-1300-5_560_2msX-75-_jpg.rf.e043be449960df1894586c890dfd503b.jpg': [3, 3, 3, 2, 4, 4, 4, 5, 5, 5, 5, 1, 1, 1], 'C-1300-5_560_2msX-25-_jpg.rf.676bc5db87a4456dbcc78011792e54bf.jpg': [4, 2, 2, 2, 2, 2, 5, 1, 5], 'B-1052-6_40-0_2msX-27-_jpg.rf.6af588db744e622b01b3984c1f0e3790.jpg': [3, 2, 2, 1, 1, 5, 3, 2, 2, 2, 4, 4, 1, 1, 1, 1, 1], 'C-1300-5_560_2msX-67-_jpg.rf.1e70e1b5c3e28b6c182a70c559428164.jpg': [4, 4, 5, 5, 4, 1, 3], 'cropped_B-1300-5_56-0_2mmX-125-_jpg.rf.b0df994ba3419b8bed5915b485f2100a.jpg': [4, 2, 2, 2, 2, 3], 'cropped_B-1052-6_40-0_2msX-51-_jpg.rf.66e468275f754401de94fdc6bdcaf15e.jpg': [5, 4, 5, 4, 1, 4, 4], 'cropped_B-1300-5_56-0_2mmX-194-_jpg.rf.760be1803fd497ab919e65b42f163bbb.jpg': [4, 3, 3, 1, 5, 5, 5, 4], 'D-1700-4_600_5msX-46-_jpg.rf.f6f87185516a15998d90a7e09b946d3d.jpg': [2, 5, 3, 3, 5, 2, 2, 2, 4, 4, 4, 4, 5, 5, 1, 1, 1, 3, 1, 1, 2, 4], 'C-1300-5_560_2msX-46-_jpg.rf.1f66ac45e3bb5652e41cc1331a1462ed.jpg': [4, 4, 5, 5, 4, 1, 3], 'cropped_B-1300-5_56-0_2mmX-125-_jpg.rf.0a7f53a6ab0fe53d0d882401551ea3c7.jpg': [1, 4, 1], 'cropped_C-868-_7140_2msX-80-_jpg.rf.1812c94132d09de79b7070a83af4b91e.jpg': [4], 'cropped_868kg0_2msX-457-_jpg.rf.b0164d8a35ff9bb8dd03fb6dd1af8d52.jpg': [3, 4, 2, 2, 4, 4, 4, 5, 3], '-0_2X-65-_jpg.rf.cae7a479f87171fc1e0c719e151fafc7.jpg': [5, 5, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 5, 5], 'D-868-7_140_2msX-196-_jpg.rf.c78ff4f97a439a78a1d3095a869c49b3.jpg': [3, 5, 2, 2, 4, 4, 4, 4, 5, 5, 5, 3, 1, 1, 1], '-0_2X-201-_jpg.rf.8043fea52a0d4fcf877c59a7eb7b0799.jpg': [3, 1, 4], 'cropped_C-1300-5_56-0_2X-22-_png.rf.505916d5a7586e661491ee578a2788ee.jpg': [4, 1], 'cropped_C-868-_7140_2msX-47-_jpg.rf.2e6ba4c40730432ab63208b95c0d5259.jpg': [3, 4, 2, 2, 4, 4, 4, 5, 3], '-0_2X-65-_jpg.rf.116a70c4eeedec28b1f44d46f017e797.jpg': [4, 3, 3, 1, 5, 5, 5, 4], 'D-1700-4_600_5msX-46-_jpg.rf.c6a74997559460f5b019f4a20b5704ed.jpg': [2, 2, 4, 5, 3], 'cropped_B-1052-6_40-0_2msX-116-_jpg.rf.327e5e86f19f6670ea51c7855fa11a96.jpg': [5, 5, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5], 'D-868-7_140_2msX-194-_jpg.rf.2390b0687acc93efc46663cb4e2fde54.jpg': [4, 2, 2, 2, 4, 4], '-0_2X-19-_jpg.rf.90013473bd8c38423fd24bb6278ebf88.jpg': [5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 5, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'D-868-7_140_2msX-195-_jpg.rf.48766857766ebbd366ce85f6831d4a94.jpg': [5, 5, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5], 'D-868-7_140_2msX-194-_jpg.rf.59ebe3ff73052e5cd3137f596a56e16e.jpg': [3, 3, 3, 3, 3, 3, 3, 3, 1, 4], 'C-1700-4_600_2mmX-41-_jpg.rf.9c41670ae8af5c2422e9a1dabdbbffb9.jpg': [5, 5, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5], 'D-868-7_140_2msX-194-_jpg.rf.53747ab56249095589fda8a3bd742eb6.jpg': [4], 'cropped_868kg0_2msX-457-_jpg.rf.15d305f247078b660ec01229f3461e80.jpg': [5, 3, 4, 2, 1, 1, 1, 3], 'cropped_B-1700-4_60-0_2X-92-_jpg.rf.14c6cabef4f5bd8e9d1116714d4caf24.jpg': [3, 3, 3, 3, 3, 3, 3, 3, 1, 4], 'C-1700-4_600_2mmX-41-_jpg.rf.3fd89e5c652deed67373da7ad9a98450.jpg': [4, 5, 5, 4, 5, 4, 1, 3, 4], 'cropped_B-1300-5_56-0_2mmX-136-_jpg.rf.87e89207a98cf72669fe7787b8eb15d4.jpg': [5, 4, 4, 4, 4, 1, 5, 5, 4], 'D-868-7_140_2msX-52-_jpg.rf.cd9e0a37ec1b8eee987afc77b2cac8df.jpg': [3, 1, 4], 'cropped_C-1300-5_56-0_2X-22-_png.rf.cdfaa0aa2196d9c0f0429ad03af378c0.jpg': [5, 5, 4, 4, 3, 3, 1, 4, 5], 'D-1700-4_600_5msX-13-_jpg.rf.17962f36ee5d15a0cd3d742bbbd78124.jpg': [5, 5, 4, 5, 4, 4, 4, 1], 'D-868-7_140_2msX-35-_jpg.rf.ec5907e7a898048f715c09c6e09991d9.jpg': [3, 4, 4, 2, 2, 1, 1, 3, 5, 5, 5, 5], 'cropped_B-1700-4_60-0_2X-186-_jpg.rf.8e22059cc80d16339eec2fc1df30c95f.jpg': [3, 3, 1, 1, 5, 4, 1], 'C-1700-4_600_2mmX-195-_jpg.rf.c51c429c964bca7f3b1881489726a406.jpg': [1, 4, 5, 4, 4, 5], 'cropped_D-868-7_140_2msX-34-_jpg.rf.08604f22a0c911f713cf849b5e128bdb.jpg': [5, 5, 4, 4, 3, 3, 1, 4, 5], 'D-1700-4_600_5msX-13-_jpg.rf.77fa28a975a205d5d215841c5335c2fc.jpg': [3, 3, 3, 3, 3, 3, 3, 3, 1, 4], 'C-1700-4_600_2mmX-41-_jpg.rf.381ab2bd082e08ef1a0f94c184d2955a.jpg': [3, 4, 2, 4, 4, 5, 5, 5, 5, 3, 1, 1, 1], '-0_2X-372-_jpg.rf.51447007cddf7796a66c07854a51eb71.jpg': [1, 1, 1, 2, 3, 3, 5, 2, 2, 2, 2, 4, 2, 4, 1, 3, 1, 1], 'C-1300-5_560_2msX-70-_jpg.rf.c7b6ffaed5aba30b69810736cefc4f00.jpg': [3, 4, 2, 4, 4, 5, 5, 5, 5, 3, 1, 1, 1], '-0_2X-372-_jpg.rf.550c6246aae0a4403bf30f4c6f39ca81.jpg': [2, 4, 2, 5, 3], 'cropped_B-1052-6_40-0_2msX-88-_jpg.rf.2f2c04856ef38fc917f0b4cc5555552c.jpg': [3, 4, 2, 2, 4, 4, 5, 5, 5, 5, 3, 1], '-0_2X-200-_jpg.rf.daf6c22ce645554efa051ab372d395bb.jpg': [1, 5, 5, 4, 4, 4, 1, 1], 'cropped_D-868-7_140_2msX-200-_jpg.rf.0fbe81b699dd9cce192eca04f813fb03.jpg': [4, 4, 2, 1, 1, 1, 1, 2, 5, 1, 1], 'C-868-_7140_2msX-145-_jpg.rf.43319e6a31202eee183147a3f1a97908.jpg': [5, 5, 4, 5, 4, 4, 4, 1], 'D-868-7_140_2msX-35-_jpg.rf.1845d95f2001d91af169435364784d8d.jpg': [4, 4, 4, 4, 4, 2, 2, 2, 3, 3, 3], 'B1330-5_56-0_2msX-56-_jpg.rf.dcd612b29b1b611adcc7f6eab9aeb4ca.jpg': [4, 5, 1], 'B-1052-6_40-0_2msX-7-_jpg.rf.8ca6200203b67fa9051a5a1b24c4cede.jpg': [4, 1], 'cropped_868kg0_2msX-344-_jpg.rf.b6ac429430fa98ac49e92aacbb1c25a6.jpg': [3, 4, 2, 2, 4, 4, 4, 5, 5, 3, 5, 5, 1, 1], '-0_2X-211-_jpg.rf.b19c206f472e4a1da84a2f0294299ad3.jpg': [4, 5, 1], 'cropped_868kg0_2msX-25-_jpg.rf.4aea92bd9d04f941bf7cf70257d2e455.jpg': [5, 4, 4, 4, 4, 1, 1, 1, 1, 1, 5, 5, 4, 4], 'D-868-7_140_2msX-68-_jpg.rf.0e4d79b382912aa1b51ce0d8da660be0.jpg': [5, 5, 4, 5, 4, 3, 3, 1], 'D-1700-4_600_5msX-48-_jpg.rf.8481283040af67877c1f3d29fc72892a.jpg': [4, 1, 1], 'cropped_868kg0_2msX-190-_jpg.rf.1e6408f907910d4f973dee102fc91f5c.jpg': [3, 3, 3, 4], 'D-1700-4_600_5msX-8-_jpg.rf.4bd27c50c1c9f40338de657964178855.jpg': [5, 1, 5, 4, 4], '868kg0_2msX-10-_jpg.rf.e95f522b2510da2d362277b306c26124.jpg': [3, 4, 5, 2, 4, 2, 3, 1], 'cropped_B-1700-4_60-0_2X-87-_jpg.rf.2912caf3535e41461abb49ebd8e53aa5.jpg': [1, 5, 4, 4, 4], 'cropped_D-868-7_140_2msX-69-_jpg.rf.7fdb68744dedb9cfd9f7dcab2b66901b.jpg': [3, 4, 4, 4, 2, 4, 4, 5, 5, 5, 1, 1, 4, 3, 5], 'C-1300-5_560_2msX-23-_jpg.rf.eca19e1aba07cfd14be09658e6faf461.jpg': [4, 5, 1], 'B-1052-6_40-0_2msX-7-_jpg.rf.b9dc6b8c6447ead53316f279800b7431.jpg': [1], 'cropped_868kg0_2msX-2-_jpg.rf.4f2e57897d70c9af0c91e438915d9aed.jpg': [1, 4, 5, 5, 2, 2, 2, 2, 3, 3, 3, 5, 1], 'B-1052-6_40-0_2msX-83-_jpg.rf.acd228d7bcd9e0e8de9d6eea0e5663ad.jpg': [5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 5, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'D-868-7_140_2msX-195-_jpg.rf.08dd3718cf359c07c933a0da7ab78982.jpg': [1, 1, 4, 4, 4, 5], 'D-868-7_140_2msX-23-_jpg.rf.6ff0677c148b6795d9024a16776202b8.jpg': [4, 4, 4, 4], 'C-1700-4_600_2mmX-18-_jpg.rf.f53790db130b5af3fdd9c5a2dee9aad7.jpg': [3, 3, 3, 2, 4, 4, 4, 5, 5, 5, 5, 1, 1, 1], 'C-1300-5_560_2msX-25-_jpg.rf.7bafcbe851576d2fe659753922583ee5.jpg': [4, 4, 4, 4], 'C-1700-4_600_2mmX-24-_jpg.rf.2bc135c79280f44e7db76d343e790d3d.jpg': [3, 2, 4, 2, 1, 1, 1, 1, 1], 'cropped_C-1300-5_56-0_2X-84-_png.rf.1741bad0d857c160102f03d39d8f41cc.jpg': [3, 3, 3, 4, 4, 4, 4, 5, 5, 2, 4, 2, 1, 1], 'C-1300-5_560_2msX-26-_jpg.rf.32097b2e8e3c6373bfd39afc0f95b7d3.jpg': [4, 2, 2, 2, 2, 1, 5], 'B-1052-6_40-0_2msX-12-_jpg.rf.f6393cce80c533a32d6aa4e10b616aa4.jpg': [4, 2, 2, 2, 2, 2, 2, 2, 5, 1], 'B-1052-6_40-0_2msX-28-_jpg.rf.e271073cbba63142a480396d2d2a8059.jpg': [1, 1, 1, 1, 1, 1, 1, 1, 4], 'C-868-_7140_2msX-9-_jpg.rf.48d30448884fedd56f83e3e25ce06103.jpg': [4, 4, 4, 4, 2, 2, 3, 3, 3, 2, 2], 'B1330-5_56-0_2msX-53-_jpg.rf.8f363c88a793f2f40a72d52e529df5d6.jpg': [3, 3, 4], 'D-1700-4_600_5msX-9-_jpg.rf.eb4de1dd9c8063ffe7e0adb07a992d1a.jpg': [4, 2, 2, 2, 2, 2, 2, 2, 5, 1], 'B-1052-6_40-0_2msX-28-_jpg.rf.0f349f5913b2f345257143f5490ba4d2.jpg': [4, 4, 5, 3], 'cropped_B-1300-5_56-0_2mmX-25-_jpg.rf.fd1de9023e35b277890b63bdbe313394.jpg': [4, 2, 1, 1, 1, 1, 1, 1, 1], 'C-868-_7140_2msX-63-_jpg.rf.dc7185b9554a2ec7e2ba3920b921b901.jpg': [5, 3, 4, 2, 2, 5, 5, 5, 3, 1], 'cropped_B-1700-4_60-0_2X-161-_jpg.rf.824e315ecd70c704d6cf93b8991bfda8.jpg': [2, 5, 4, 4, 1, 5], '868kg0_2msX-482-_jpg.rf.4a379e1d709edc5de5e9bf40dbe12a4f.jpg': [5, 1, 5, 4, 4], '868kg0_2msX-10-_jpg.rf.14b1bcc7a9695484c5d48ddf13f6d9e8.jpg': [1, 1, 1, 1, 1, 1, 1, 1, 4], 'C-868-_7140_2msX-9-_jpg.rf.b175248942186400cc0c6aadc8240b8d.jpg': [3, 3, 3, 4, 4, 4, 4, 5, 5, 2, 4, 2, 1, 1], 'C-1300-5_560_2msX-26-_jpg.rf.7e7483b2b82a78e7e60715ca3b7d62f1.jpg': [5, 4, 4, 4, 4, 4, 4, 1, 4, 5, 5], 'D-868-7_140_2msX-107-_jpg.rf.2c397c2e507e1e7befc25f3d4cefcce4.jpg': [4, 2, 1, 1], 'C-868-_7140_2msX-65-_jpg.rf.6d5a8dbc6f994c3cc7020d9e9886d5b2.jpg': [4, 4, 2, 1, 1, 1, 1, 2, 5, 1, 1], 'C-868-_7140_2msX-145-_jpg.rf.3f1553872c04f9d2081e3f18599f9145.jpg': [5, 4, 4, 4, 4, 4, 4, 1, 4, 5, 5], 'D-868-7_140_2msX-107-_jpg.rf.41abf86f384bab2cbef65977f59858ef.jpg': [4, 4, 4, 4, 1, 1], 'D-868-7_140_2msX-22-_jpg.rf.56a4135430bf070970d8e76b682316e5.jpg': [4, 1], 'cropped_C-868-_7140_2msX-22-_jpg.rf.b7ffeada58de6d012403b734ea1dfe05.jpg': [1, 4, 5, 5, 2, 2, 2, 2, 3, 3, 3, 5, 1], 'B-1052-6_40-0_2msX-83-_jpg.rf.9d25133f7b167d7bf6d92e6a3ba01cdd.jpg': [4, 4, 4, 4], 'C-1700-4_600_2mmX-24-_jpg.rf.7a9014580d95a3cb30064383365b15d8.jpg': [4, 1, 1], 'cropped_C-868-_7140_2msX-35-_jpg.rf.2760170e72093c62610680128e169a7c.jpg': [3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 4, 2, 2, 4], 'C-1300-5_560_2msX-22-_jpg.rf.4b8249dc9d5367d98ca2656b3e6b3000.jpg': [5, 4, 4, 4, 4, 1, 5, 5, 4], 'D-868-7_140_2msX-52-_jpg.rf.7f557cb06c1e0f48e851b6c269b3378d.jpg': [3, 2, 4, 2, 1, 1, 1, 1, 1], 'cropped_C-1300-5_56-0_2X-84-_png.rf.4aae4a2299074840ca53d59db7cb8bd4.jpg': [4, 1], 'cropped_C-868-_7140_2msX-59-_jpg.rf.90c04f58fe3849ea743ac49b43c0d561.jpg': [3, 2, 3, 3, 3, 3, 3, 2, 2, 4, 1, 1, 1, 1, 4, 1, 1, 1, 5, 5], 'C-1300-5_560_2msX-108-_jpg.rf.5746c5c281b610cdb355621f464efbff.jpg': [3, 2, 2, 4, 4, 5, 3, 1, 1, 1, 1], '-0_2X-168-_jpg.rf.5ef801e07ad2d3231995e0f26fac3bfd.jpg': [4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1], 'C-868-_7140_2msX-67-_jpg.rf.1040bbb8806c6924d1d29d80ef34c20c.jpg': [5, 4, 4, 5, 4, 1, 1, 5], 'D-868-7_140_2msX-108-_jpg.rf.0939cc7c74665ab2766953aa92191777.jpg': [4, 4, 4, 4, 4, 2, 2, 2, 3, 3, 3, 3, 3, 2], 'B1330-5_56-0_2msX-55-_jpg.rf.7e935cd519e7222cf74526eda15df2ce.jpg': [4], '-0_2X-17-_jpg.rf.c955a88f6be973a131951d7d803b0240.jpg': [4, 4, 4, 4, 4, 1, 5, 3], 'cropped_B-1300-5_56-0_2mmX-92-_jpg.rf.985a1d7a6cc5812cf38c59002779d32a.jpg': [4, 5, 4, 3, 4, 2, 2, 3, 1, 5, 5, 5], '-0_2X-210-_jpg.rf.387105cb82d1e570f4b66e6837e1b626.jpg': [5, 4, 4, 4, 4, 1, 4], 'D-868-7_140_2msX-78-_jpg.rf.0edad0ad7ddc616254f670fd14cb8b0b.jpg': [3, 4, 2, 2, 1, 1, 1, 1], 'cropped_C-1300-5_56-0_2X-148-_png.rf.ac7b6d8422fdfcac797e4eac9a142f02.jpg': [4, 4, 5, 3], 'cropped_B-1300-5_56-0_2mmX-25-_jpg.rf.8253248b824a024806f6a05830158d49.jpg': [5, 4, 4, 4, 4, 4, 1, 1, 5], 'D-868-7_140_2msX-39-_jpg.rf.0ebfb806df43ccaee3317631e3dc6768.jpg': [4, 2, 2, 2, 2, 2, 2, 1, 1, 5], 'B-1052-6_40-0_2msX-30-_jpg.rf.750b81dbed9fd0da0754bf6c723a1a1e.jpg': [3, 3, 3, 5, 4], 'cropped_D-1700-4_600_5msX-12-_bmp.rf.abf9ae0878f4876943fa37eda2492251.jpg': [1, 1, 1, 4, 3, 3], 'C-1700-4_600_2mmX-74-_jpg.rf.b0f649b3a2bb85a42eb4846c90302b5b.jpg': [4, 4, 4, 4, 1, 5], 'D-868-7_140_2msX-13-_jpg.rf.baed006456f4dfcb9f5f5e5864a42277.jpg': [1, 4, 5, 4, 4, 5], 'cropped_D-868-7_140_2msX-34-_jpg.rf.6e742c15219b163e9d6f0fb76d8be784.jpg': [4, 5, 3, 2, 5, 5, 5, 1, 3], 'cropped_B-1700-4_60-0_2X-340-_jpg.rf.9a75a53e9f788d6d1ecac40afa2285f6.jpg': [4, 4, 4, 4, 4, 5, 1], 'C-1700-4_600_2mmX-27-_jpg.rf.723b96521703dc74e3549db171ea27a0.jpg': [2, 2, 2, 2, 2, 5, 3, 3, 4, 4, 1, 1, 3, 3, 1, 1, 1, 1], 'C-1300-5_560_2msX-106-_jpg.rf.27fe9447765cb59f918d324b994f1447.jpg': [3, 3, 4], 'D-1700-4_600_5msX-9-_jpg.rf.d4f13ef2479c4eee058e5fbb39b822ca.jpg': [1, 3], 'cropped_C-1700-4_600_2mmX-63-_jpg.rf.0c1f4e0f9053150dbb1344affd645bbb.jpg': [4, 1, 2, 5, 2], 'B-1052-6_40-0_2msX-9-_jpg.rf.05d603891f37144a3dd1d31bac7d809a.jpg': [3, 3, 1, 3, 4, 1, 1], 'C-1700-4_600_2mmX-44-_jpg.rf.b7170549db7f501662bc7fcfc95a022e.jpg': [4, 4, 1, 1], 'cropped_C-868-_7140_2msX-113-_jpg.rf.6dbb3f52fb77405c316e7791fed2c833.jpg': [3, 5, 3, 1, 4], 'C-1700-4_600_2mmX-197-_jpg.rf.0c3cf5ac94035bdadf407f44d9c9ffdc.jpg': [2, 2, 2, 1, 1, 1, 5, 5, 4, 4, 3, 3, 5, 1, 1, 1, 1, 4, 3], 'C-1300-5_560_2msX-109-_jpg.rf.5a58f38e478258a62fc5292ad5c3ea11.jpg': [4, 4, 4, 5], 'cropped_B-1300-5_56-0_2mmX-13-_jpg.rf.73adbdf300cd1035d8ed8af6e722611d.jpg': [4, 2, 2, 4, 1], 'cropped_C-868-_7140_2msX-100-_jpg.rf.be58527fd40918904b02185a365688c7.jpg': [5, 4, 4, 4, 4, 4, 1, 1, 5], 'D-868-7_140_2msX-39-_jpg.rf.008709c0c2ddefab25987a23900cbeb2.jpg': [4, 4, 4, 4, 1, 1], 'D-868-7_140_2msX-22-_jpg.rf.c37e23ecf70e809f5b810ecb983c0565.jpg': [4, 1], 'cropped_C-868-_7140_2msX-70-_jpg.rf.d779346334bad98066274d4cd9bda42c.jpg': [4, 4, 1, 1], 'cropped_C-868-_7140_2msX-113-_jpg.rf.ebe9ce4170408aff6f8be007774d02ef.jpg': [1, 4, 1], 'cropped_C-868-_7140_2msX-80-_jpg.rf.9a3ef44752a6c604e904314dd9dc45c5.jpg': [3, 2, 2, 4, 4, 4, 5, 5, 4, 3, 1], '-0_2X-212-_jpg.rf.5d5aca30696efe6c3e33738c415cab8f.jpg': [1, 4, 4, 4, 4], 'C-868-_7140_2msX-8-_jpg.rf.97043549e83998b191ef25c61385aa83.jpg': [3, 2, 2, 1, 1, 5, 3, 2, 2, 2, 4, 4, 1, 1, 1, 1, 1], 'C-1300-5_560_2msX-67-_jpg.rf.c2c4f84be0b10f7346bf3a611dac5c8c.jpg': [3, 3, 1, 1, 1, 1, 1, 1, 4], 'C-1700-4_600_2mmX-47-_jpg.rf.1b0d7a5a754dcbaf12bd375e9fa11d41.jpg': [1, 4, 1, 3, 3], 'C-1700-4_600_2mmX-71-_jpg.rf.85eaf95486ecfbef6feb606defc44783.jpg': [4, 4, 2, 3, 1, 1, 5, 5], 'cropped_B-1300-5_56-0_2mmX-113-_jpg.rf.5d1a6d4e7d0d1148e7e0c4064e755cfc.jpg': [1, 5, 4, 5, 2], '868kg0_2msX-403-_jpg.rf.75a87398780d7d86a312e8a1bac57936.jpg': [3, 2, 2, 2, 2, 4, 4, 5], 'cropped_B-1700-4_60-0_2X-40-_jpg.rf.e64d094b56d924f3dbf623eb3b495d5a.jpg': [3, 1, 1, 1], 'cropped_C-1700-4_600_2mmX-197-_jpg.rf.598f4fe3d399d8347668c8bb8d17e3e0.jpg': [1, 5, 4, 4, 4], 'cropped_D-868-7_140_2msX-69-_jpg.rf.c59e5c45df5e6636bc3e9f4f22c92dd8.jpg': [5, 4, 4, 4, 5, 5, 5, 5, 1, 1], '868kg0_2msX-16-_jpg.rf.7694360cef67b7dbc6f365e2ace960ab.jpg': [4, 2, 2, 4, 1], 'cropped_C-868-_7140_2msX-106-_jpg.rf.5839c805de3c097b2aa856cc3f913eed.jpg': [5, 5, 5, 5, 4, 5, 4, 4, 4, 1, 1], 'D-868-7_140_2msX-76-_jpg.rf.1e4650e9b09bf366d37595f4e5fb11a4.jpg': [4, 4, 1, 4], 'cropped_C-868-_7140_2msX-135-_jpg.rf.71e2d0b87c4b537a04f939d97cb9ce11.jpg': [5, 5, 4, 4, 4, 4, 1, 1, 1], 'D-868-7_140_2msX-72-_jpg.rf.16c57c460eac570dbfde14c460beb623.jpg': [5, 4, 4, 1, 5, 2], '868kg0_2msX-405-_jpg.rf.b8e9412a7639079179f876f32f4ecfea.jpg': [5, 4, 4, 4, 1, 5], '868kg0_2msX-8-_jpg.rf.56aeece0c9fdcea24a87ce791f7d5b03.jpg': [2, 5, 3, 3, 5, 2, 2, 2, 4, 4, 4, 4, 5, 5, 1, 1, 1, 3, 1, 1, 2, 4], 'C-1300-5_560_2msX-46-_jpg.rf.1d60b07c448e72e32a19f0d4f8182e3a.jpg': [4, 2, 1, 1, 1, 1, 1, 1, 1], 'C-868-_7140_2msX-63-_jpg.rf.78af66476f93459a6235a3e8de2d812f.jpg': [5, 4, 5, 4, 1, 4, 4], 'cropped_B-1300-5_56-0_2mmX-194-_jpg.rf.7eaa997125b36d5e701c547ae3a13b79.jpg': [2, 2, 2, 1, 1, 1, 5, 5, 4, 4, 3, 3, 5, 1, 1, 1, 1, 4, 3], 'C-1300-5_560_2msX-109-_jpg.rf.0b241f222dc5d60343adbe742b5dd1c3.jpg': [2, 2, 4, 3, 1], 'cropped_B-1052-6_40-0_2msX-126-_jpg.rf.115a107a11f9cb6d8a8172dcf229e02f.jpg': [4, 2, 2, 2, 2, 3], 'cropped_B-1052-6_40-0_2msX-51-_jpg.rf.5630d474279be062139fff9bab8ea81a.jpg': [5, 3, 4, 2, 1, 1, 1, 3], 'cropped_B-1700-4_60-0_2X-92-_jpg.rf.d41ad42f549c97f67165564397ce0d16.jpg': [3, 4, 2, 2, 4, 4, 5, 5, 5, 3, 1, 1], '-0_2X-163-_jpg.rf.309422b89778d4a391ffa675a7fea589.jpg': [4, 4, 4, 4, 2, 2, 3, 3, 3, 2, 2], 'B1330-5_56-0_2msX-53-_jpg.rf.a04689089dabcb8b698e0a9371a3ef7a.jpg': [3, 2, 3, 3, 3, 3, 3, 2, 2, 4, 1, 1, 1, 1, 4, 1, 1, 1, 5, 5], 'C-1300-5_560_2msX-108-_jpg.rf.d2b55c6f59ca6dddd2e86fe67c428a1e.jpg': [4, 5, 1], 'B-1052-6_40-0_2msX-7-_jpg.rf.e6568aefd37499a6745fd5172236b65e.jpg': [4], '-0_2X-17-_jpg.rf.9af1b663a8a57042002479758214af09.jpg': [4, 2, 2, 2, 2, 2, 2, 2, 5, 1], 'B-1052-6_40-0_2msX-28-_jpg.rf.74a6f4bec34a84121a97c5a174f79475.jpg': [3, 1], 'cropped_C-1700-4_600_2mmX-115-_jpg.rf.d2a73a4b8096246ebd1df9391fc8693a.jpg': [4, 4, 4, 4, 4, 2, 2, 2, 3, 3, 3, 3, 3, 2], 'B1330-5_56-0_2msX-55-_jpg.rf.5c222dfefbfa349d5387f86c97501f09.jpg': [2, 2, 4, 1, 5, 3], 'cropped_B-1052-6_40-0_2msX-100-_jpg.rf.56aa72eadef9b1bd4f3294bcaa1d394f.jpg': [5, 4, 4, 1, 5, 2], '868kg0_2msX-405-_jpg.rf.c0bbf8d5043fdad745a4e7065cf01cb8.jpg': [3, 1, 1, 4, 3, 3, 2], 'C-1300-5_560_2msX-169-_jpg.rf.0f7c9fbef877c49597c4778c354597cc.jpg': [5, 4, 2, 1, 5], '868kg0_2msX-481-_jpg.rf.34a6a34cdaa7cad083963599fa321798.jpg': [5, 4, 4, 4, 5, 5, 5, 5, 1, 1], '868kg0_2msX-16-_jpg.rf.387532008b1a49c332b7fe99767a9d99.jpg': [1, 4, 4], 'cropped_C-868-_7140_2msX-124-_jpg.rf.15d5e65570ecd60499aeb677ebb646b9.jpg': [4, 4, 4, 4, 4, 1, 5, 3], 'cropped_B-1300-5_56-0_2mmX-92-_jpg.rf.c2b01bf79b61895a66d1005349124b64.jpg': [5, 4, 4, 4, 4, 4, 3, 4, 5, 4], 'D-1700-4_600_5msX-12-_jpg.rf.5dd78bf29391dbb7d17b397955d77959.jpg': [1, 5, 4, 5, 2], '868kg0_2msX-403-_jpg.rf.e05c49b70acad99c7de4ac225c15798a.jpg': [5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 5, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'D-868-7_140_2msX-195-_jpg.rf.5df4c3801280811aae69689bfd1423e9.jpg': [3, 1], 'cropped_C-1700-4_600_2mmX-160-_jpg.rf.43e4eec9d78849ddf526ba3a1f912b3d.jpg': [2, 5, 3, 2, 2, 2, 2, 2, 2, 4, 4, 1, 3, 3, 1, 1, 1, 1], 'C-1300-5_560_2msX-64-_jpg.rf.31dfcfc10ca5e6b10975d50a17e1b15e.jpg': [3, 5, 2, 2, 4, 4, 4, 4, 5, 5, 5, 3, 1, 1, 1], '-0_2X-201-_jpg.rf.1c6d2e75b8b767c4bc2bf973ce498e50.jpg': [3, 4, 2, 2, 4, 4, 5, 5, 5, 5, 5, 3, 1], '-0_2X-197-_jpg.rf.dd41a3fb6739a26523ec908685c1f216.jpg': [4, 1, 1, 2, 2], 'C-868-_7140_2msX-34-_jpg.rf.96d89189b33f50fc5b519f04071d6660.jpg': [4], 'cropped_868kg0_2msX-465-_jpg.rf.85392cea7c0a95047fd583c18a785dc8.jpg': [5, 5, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 5, 5], 'D-868-7_140_2msX-196-_jpg.rf.4b065d1e0fa313fe26e744fd7f07eeaf.jpg': [5, 4, 4, 4, 4, 4, 1, 1, 1, 5, 4], 'D-868-7_140_2msX-106-_jpg.rf.0ad0452851858f990249fd537c9562d8.jpg': [4, 3, 1, 5, 5, 5, 4], 'D-1700-4_600_5msX-49-_jpg.rf.008f219135f270ef5fefc5110c82f4f8.jpg': [4, 4, 4, 4, 2, 2, 4, 3, 3, 3], 'B1330-5_56-0_2msX-90-_jpg.rf.8f4b20c4276405eaaec50b1498266a15.jpg': [3, 2, 2, 2, 2, 4, 4, 5], 'cropped_B-1700-4_60-0_2X-40-_jpg.rf.21497bd2e4137e746d5d663c11ee016d.jpg': [3, 4, 4, 2, 2, 4, 4, 5, 5, 5, 5, 5, 5, 3, 1, 1], '-0_2X-209-_jpg.rf.91fc726d7789dfb5a8780f3acf426a78.jpg': [4, 2, 2, 5, 4], 'B1330-5_56-0_2msX-7-_jpg.rf.c34b0200f66fea9802b86d174cfc6d5b.jpg': [4, 4, 4, 4, 4, 5, 1], 'C-1700-4_600_2mmX-27-_jpg.rf.0ca1b4160a0cc037e10192a4816b7490.jpg': [4, 4, 2, 2, 2, 2], 'cropped_B1330-5_56-0_2msX-13-_jpg.rf.dcc73bba4eed18d528e14aa1d4efdab9.jpg': [3, 3, 3, 2, 4, 4, 4, 5, 5, 5, 5, 1, 1, 1], 'C-1300-5_560_2msX-25-_jpg.rf.3ed649883e96581c07ed5bdb1749110a.jpg': [5, 4, 4, 2, 5, 1, 1], '868kg0_2msX-129-_jpg.rf.3bdf9536bc6076a3167d56f3b7f83c57.jpg': [4, 4, 1, 4], 'cropped_C-868-_7140_2msX-135-_jpg.rf.c47589beec1a40996a2315d29d7c8486.jpg': [5, 4, 4, 2, 5, 1, 1], '868kg0_2msX-129-_jpg.rf.15cb6fbdbd2e12f4ac91990c792fbde1.jpg': [4, 4, 1], 'cropped_868kg0_2msX-256-_jpg.rf.680abaea240c8f96ca6ddf5c5cfdd053.jpg': [4, 1, 2, 5, 2], 'B-1052-6_40-0_2msX-9-_jpg.rf.d7997ca122f19b6409614dd85a855ddb.jpg': [3, 4, 1], 'cropped_C-1700-4_600_2mmX-24-_jpg.rf.f074f9d9307f7bc1dd415e0918b513fd.jpg': [5, 4, 4, 4, 3, 3, 4, 5, 4], 'D-1700-4_600_5msX-11-_jpg.rf.2f1448dbc89d35a1a83579500800de61.jpg': [1, 4, 4], 'cropped_C-868-_7140_2msX-124-_jpg.rf.aaec3db4604ccce003a8f2300d204bee.jpg': [4, 4, 2, 1, 1, 1, 2, 1], 'C-868-_7140_2msX-148-_jpg.rf.565a809ea9e9e483e244954cc410da17.jpg': [5, 4, 4, 4, 4, 2, 2, 4, 4, 3, 3, 3], 'B1330-5_56-0_2msX-89-_jpg.rf.fd760ee135ca5fb8e6cb46b45b0c4968.jpg': [4, 4, 4, 4, 5, 4, 1], 'D-868-7_140_2msX-10-_jpg.rf.d1893df6877a6e720f8ec1fe73e2fb20.jpg': [4, 5, 4, 3, 4, 2, 2, 3, 1, 5, 5, 5], '-0_2X-210-_jpg.rf.ca0215422bbc94fd466ab5c3bc7c0db8.jpg': [1], 'cropped_D-868-7_140_2msX-0-_jpg.rf.af8c9fe2fc05d6bc7ad168b80d3f5da9.jpg': [4, 4, 4, 4], 'B1330-5_56-0_2msX-9-_jpg.rf.f07c72378aeedafe4f9cb5f3b70f6c63.jpg': [4, 4, 5, 4, 4, 1, 1, 4, 4, 1, 4, 4, 4, 4], 'D-868-7_140_2msX-40-_jpg.rf.8fa2471c9a462ae496829c914b181cb5.jpg': [5, 4, 5, 5, 1, 2, 2], '868kg0_2msX-128-_jpg.rf.709ff8f522d20b80e17f79a0c22bc8d2.jpg': [5, 5, 4, 4, 4, 4, 1, 1, 1], 'D-868-7_140_2msX-72-_jpg.rf.0932db0f0b57f4ba561e9e9483f5c908.jpg': [2, 2, 2, 3, 1, 1, 4, 1, 1, 1, 3, 4, 5, 1], 'C-1300-5_560_2msX-74-_jpg.rf.c74a60119859d5acdcf2cec0505115d7.jpg': [5, 4, 4, 5, 4, 1, 1, 5], 'D-868-7_140_2msX-108-_jpg.rf.d49889877efc46195cc6aa94e9cc3f8c.jpg': [3, 5, 4, 3, 1, 1, 1, 1], 'C-1700-4_600_2mmX-196-_jpg.rf.e21391c9512ee68814a2bb0370be7413.jpg': [1, 3], 'cropped_C-1700-4_600_2mmX-63-_jpg.rf.da5a71064f2ca41a2643c7a5939b8070.jpg': [3, 1, 1, 1, 5, 2, 3, 5, 2, 2, 2, 2, 2, 4, 4, 4, 5, 1, 1, 1], 'C-1300-5_560_2msX-47-_jpg.rf.27e4c35453074768ec5da869909c600d.jpg': [4, 4, 5, 4, 4, 1, 1, 4, 4, 1, 4, 4, 4, 4], 'D-868-7_140_2msX-40-_jpg.rf.a14c391fcc5dc10f890b12dcaafa0ed8.jpg': [5, 5, 5, 5, 5, 4, 4, 4, 4, 1, 1, 1], 'D-868-7_140_2msX-70-_jpg.rf.1290249e11c4c024cd4616f7554ec299.jpg': [2, 2, 4, 3, 1], 'cropped_B-1052-6_40-0_2msX-126-_jpg.rf.52b4118380118e29f58302a93248637f.jpg': [4], 'cropped_868kg0_2msX-465-_jpg.rf.a61bcd7c304dc5ee370f4595d83df0e8.jpg': [1, 4, 4, 4, 5], 'D-868-7_140_2msX-26-_jpg.rf.9abd92edb537fcfc7baf3868a982a004.jpg': [4, 4, 4, 2, 2, 3, 3, 2], 'cropped_B1330-5_56-0_2msX-25-_jpg.rf.2981f36394350068e5e760ff3e913d9c.jpg': [1], 'cropped_D-868-7_140_2msX-0-_jpg.rf.2b6f6263a7d9784e6b5bfe24cdad18bc.jpg': [5, 5, 4, 3, 3], 'D-1700-4_600_5msX-10-_jpg.rf.9a4d18bfce5d1bdd1f65815de5573997.jpg': [3, 4, 2, 2, 1, 1, 1, 1], 'cropped_C-1300-5_56-0_2X-148-_png.rf.acba7d393e883322e3792738f70c8cf9.jpg': [4, 4, 1, 1, 4, 1, 1], 'cropped_C-868-_7140_2msX-102-_jpg.rf.1f69b4f0b21e3f0ddd3f440ce9408525.jpg': [4, 4, 2, 4, 2, 1, 1, 1, 1, 1], 'C-868-_7140_2msX-31-_jpg.rf.ec3934d7ba26e18abb1d1b43b6d7412c.jpg': [3, 2, 2, 5, 4, 3, 3, 3, 1, 1, 1, 1, 5, 5, 5, 5], 'cropped_B-1700-4_60-0_2X-315-_jpg.rf.0c8c274d7dbfa40960e95d2e6ff805b3.jpg': [1, 1, 1, 4, 3, 3], 'C-1700-4_600_2mmX-74-_jpg.rf.b00c4623460c9045c0fd3e45560723f1.jpg': [3, 3, 3, 5, 4], 'cropped_D-1700-4_600_5msX-12-_bmp.rf.40c6aad76461b7ce9804721a3822caef.jpg': [4, 4, 4, 2, 2, 3, 3, 2], 'cropped_B1330-5_56-0_2msX-25-_jpg.rf.1bf229723d64f06dc06869046c72f8a6.jpg': [3, 4, 4], 'cropped_C-1300-5_56-0_2X-35-_png.rf.2f91abde9dcfd9a9bc93d4222e6cb2b7.jpg': [5, 3, 4, 2, 2, 5, 5, 5, 3, 1], 'cropped_B-1700-4_60-0_2X-161-_jpg.rf.28212eaba14e3fc8411673ec7077845e.jpg': [3, 3, 4, 2, 2, 4, 4, 5, 5, 5, 3, 1], '-0_2X-167-_jpg.rf.b8e390404e786326e270187e2d341496.jpg': [3, 4, 4, 2, 2, 4, 4, 5, 5, 5, 5, 5, 5, 3, 1, 1], '-0_2X-209-_jpg.rf.f1e191c7b5ea0630223745d9611b944f.jpg': [4, 4, 4, 4, 1, 5], 'D-868-7_140_2msX-13-_jpg.rf.cb4aa1b794346a27237fba883214735a.jpg': [3, 4, 2, 2, 4, 4, 5, 5, 5, 5, 5, 3, 1], '-0_2X-197-_jpg.rf.bb31afa5d35a7d9e64272b3b2d9706eb.jpg': [1, 4, 4, 4, 4], 'C-868-_7140_2msX-8-_jpg.rf.75c89b0294bdb2d35c9ba13c61ba2032.jpg': [4, 4, 4, 4, 4, 4, 4, 5, 4, 1], 'D-868-7_140_2msX-11-_jpg.rf.3e46b58d918b7328c01eb78ac57934e5.jpg': [5, 4, 2, 4, 3, 1, 1, 1, 3, 1], 'cropped_B-1700-4_60-0_2X-374-_jpg.rf.503c00eb2625446a00f789dd0f6ac82b.jpg': [3, 1, 1, 2, 4, 2, 2], 'cropped_C-1300-5_56-0_2X-127-_png.rf.eb09a3b4cac31805c3e63c1f1c6c2b34.jpg': [5, 5, 5, 5, 4, 5, 4, 4, 4, 1, 1], 'D-868-7_140_2msX-76-_jpg.rf.1923eedde22b117771cb6aeb85db6406.jpg': [3, 5, 4, 3, 1, 1, 1, 1], 'C-1700-4_600_2mmX-196-_jpg.rf.524a6ffcefb4bec3672db62b4a82b164.jpg': [4, 4, 2, 1, 1], 'C-868-_7140_2msX-151-_jpg.rf.47586b9ccf3c64c732083dea8c64a823.jpg': [3, 4, 4], 'cropped_C-1300-5_56-0_2X-35-_png.rf.a097853c8dfc2954efabc88d70b66299.jpg': [5, 4, 4, 4, 4, 4, 3, 4, 5, 4], 'D-1700-4_600_5msX-12-_jpg.rf.c2a237f9e1cdb61e6071f268ab51eba1.jpg': [3, 4, 2, 2, 4, 4, 5, 5, 5, 3, 1, 1], '-0_2X-163-_jpg.rf.2f71a6676455b5f695371cc680da910b.jpg': [5, 5, 5, 5, 4, 4, 4, 1, 1, 4, 5], 'D-868-7_140_2msX-105-_jpg.rf.07744cd73433f0f820b7c7710c210548.jpg': [4, 4, 2, 2, 2, 3, 3, 3], 'cropped_B1330-5_56-0_2msX-48-_jpg.rf.cc3aaa3c29142bdc32b1d69cd3e5ceec.jpg': [3, 1, 4], 'cropped_C-1700-4_600_2mmX-44-_jpg.rf.a6abf53a4b73abdab2ff3a655c95e7bb.jpg': [5, 4, 3, 3, 5, 5, 4], 'D-1700-4_600_5msX-14-_jpg.rf.95967c182e1c3aa9a75887e03808eb57.jpg': [4, 4, 4, 3], 'cropped_B-1300-5_56-0_2mmX-49-_jpg.rf.e00d6d783c98f3db32bae82d96becacb.jpg': [4, 4, 2, 2, 2, 3, 3], 'cropped_B1330-5_56-0_2msX-92-_jpg.rf.963121958b9f952cac317ec88ce587f4.jpg': [4, 4, 2, 1, 1, 1, 2, 1], 'C-868-_7140_2msX-148-_jpg.rf.4cf864647e0a18341ed8bb0181f6d5b8.jpg': [5, 4, 4, 5, 2, 2, 3, 1, 1, 3, 1], 'cropped_B-1700-4_60-0_2X-375-_jpg.rf.dc877911a93432c4d7d1f36c6af4ba10.jpg': [1, 3, 5, 2, 2, 2, 2, 2, 4, 4, 1, 5, 1, 1, 2, 2, 3, 3, 3, 3, 3, 1, 1, 1], 'C-1300-5_560_2msX-72-_jpg.rf.8206445dd62d9c2e1c20f5b94dd80e62.jpg': [4, 3, 4, 2, 3, 2, 2], 'cropped_B1330-5_56-0_2msX-71-_jpg.rf.dae444d8b9a5dcd4854a30e18a919608.jpg': [4, 1], 'cropped_C-868-_7140_2msX-47-_jpg.rf.79616e6af58a87120ad1a2e9e0a05eda.jpg': [1, 1, 2, 5, 3, 2, 5, 2, 2, 2, 4, 2, 4, 4, 4, 1, 3, 1, 1, 1, 3, 3], 'C-1300-5_560_2msX-63-_jpg.rf.26421289235a9ca9bb7a844bdc01c643.jpg': [5, 4, 2, 4, 3, 1, 1, 1, 3, 1], 'cropped_B-1700-4_60-0_2X-374-_jpg.rf.5d59a18f2c9a0993fd982b2b2440ebbf.jpg': [1], 'cropped_868kg0_2msX-2-_jpg.rf.92ce69020b6024474540add481c3baa3.jpg': [3, 2, 2, 2, 2, 4, 1, 1, 1], 'cropped_C-1300-5_56-0_2X-72-_png.rf.a1c363d2b27f2b84a927f2dd760ae5f6.jpg': [1, 1, 4, 2, 2, 2, 2, 3], 'cropped_C-1300-5_56-0_2X-60-_png.rf.91e40a792ea24940da519dd447f7b02b.jpg': [5, 4, 3, 3, 2, 2, 5, 5, 1, 1, 1, 5], 'B-1052-6_40-0_2msX-170-_jpg.rf.fcafb4f96905ee6861c4908e5de4bb1f.jpg': [4, 2, 2, 4, 1], 'cropped_C-868-_7140_2msX-100-_jpg.rf.51b108f7e71e653a69785d57b7cf9e2d.jpg': [1, 4, 4, 4, 5], 'D-868-7_140_2msX-26-_jpg.rf.5758095a40c635efac7b46f05835634a.jpg': [3, 1, 1, 4, 3, 3, 2], 'C-1300-5_560_2msX-169-_jpg.rf.1388c35d658209d1a18192ded8e91f4a.jpg': [4, 3, 4, 2, 3, 2, 2], 'cropped_B1330-5_56-0_2msX-71-_jpg.rf.aad27601a4c83e3a1ad7a2d1e621e862.jpg': [1, 4, 5, 5, 2, 2, 2, 2, 3, 3, 5, 5, 1], 'B-1052-6_40-0_2msX-85-_jpg.rf.58419650ec373eb8abe71a4e017fe8e2.jpg': [2, 2, 2, 3, 1, 1, 4, 1, 1, 1, 3, 4, 5, 1], 'C-1300-5_560_2msX-74-_jpg.rf.6c4c31f66c7f90b2f95ff0c550277678.jpg': [3, 1], 'cropped_C-1700-4_600_2mmX-160-_jpg.rf.ddc5e7031d7f9bc1db9d70a133429e14.jpg': [3, 3, 4, 2, 2, 4, 4, 5, 5, 5, 3, 1], '-0_2X-167-_jpg.rf.75b3eaf0473fc5282df27afb5be9e26b.jpg': [5, 4, 4, 4, 3, 3, 4, 5, 4], 'D-1700-4_600_5msX-11-_jpg.rf.ff9f061d7beed0849ef7338362da926f.jpg': [5, 4, 4, 4, 4, 1, 1, 5, 1], '868kg0_2msX-17-_jpg.rf.0cbfeca613183a378e46c7b0a2dd1de1.jpg': [1, 4, 4], 'cropped_C-868-_7140_2msX-124-_jpg.rf.5ac04fd273189384702df2674af72f76.jpg': [3, 2, 2, 2, 2, 4, 1, 1, 1], 'cropped_C-1300-5_56-0_2X-72-_png.rf.7db892ef650a2b93837c00ccabc66bf8.jpg': [4, 4, 4, 4, 4, 4, 4, 5, 4, 1], 'D-868-7_140_2msX-11-_jpg.rf.e8674b4fe9ba4dc0bb73b0ec9edb0aec.jpg': [1, 1, 4, 2, 2, 2, 2, 3], 'cropped_C-1300-5_56-0_2X-60-_png.rf.8664f52bfb9ef8df01f7d7662fc7118f.jpg': [4, 2, 2, 1, 2], 'B-1052-6_40-0_2msX-11-_jpg.rf.50423e3ec6e1d1ce93447a2130579c8a.jpg': [1, 3, 5, 2, 2, 2, 2, 2, 4, 4, 1, 5, 1, 1, 2, 2, 3, 3, 3, 3, 3, 1, 1, 1], 'C-1300-5_560_2msX-72-_jpg.rf.52e37e6c82de9d6529bcf48fed340053.jpg': [3, 1, 4], 'cropped_C-1700-4_600_2mmX-44-_jpg.rf.5cc1686522213b9a90afa214d9225d6d.jpg': [3, 4, 2, 2, 2, 4, 4, 5, 5, 5, 3, 5, 1, 1, 1, 1, 1], '-0_2X-164-_jpg.rf.1e057eafbfc1a619d9a492546c9ba926.jpg': [3, 4, 4, 2, 2, 2, 2, 2, 2], 'cropped_C-1300-5_56-0_2X-47-_png.rf.aa754dd18d98ec687293c57506819fc8.jpg': [3, 1], 'cropped_C-1700-4_600_2mmX-186-_jpg.rf.664caebd3493bbb4d4c87fac29615733.jpg': [2, 4, 2, 4, 4, 5, 5, 3], '-0_2X-16-_jpg.rf.c40f04b77da485a7b443a327e18dead2.jpg': [3, 4, 4, 2, 2, 2, 2, 2, 2], 'cropped_C-1300-5_56-0_2X-47-_png.rf.4119c714609d241da893e7b8e6333b62.jpg': [5, 4, 3, 3, 2, 2, 5, 5, 1, 1, 1, 5], 'B-1052-6_40-0_2msX-170-_jpg.rf.f3d46dbe83927cabade96fe2b57278bd.jpg': [4, 4, 5, 5, 5, 1, 4, 3], 'cropped_B-1300-5_56-0_2mmX-157-_jpg.rf.05cbe80ed54218857c3351c51adce60f.jpg': [2, 4, 2, 3, 3, 1, 1, 1], 'cropped_C-1300-5_56-0_2X-173-_png.rf.5757a2411de81f7fcd616a944619525e.jpg': [2, 4, 4, 1, 1, 1, 1, 1, 1, 1, 2], 'C-868-_7140_2msX-104-_jpg.rf.5b5a16a8a60f57aa1e8de17f500d265a.jpg': [3, 1, 1, 2, 4, 2, 2], 'cropped_C-1300-5_56-0_2X-127-_png.rf.2da57fbfc39933fd5ff6d78ffb8588b3.jpg': [5, 4, 3, 3, 5, 5, 4], 'D-1700-4_600_5msX-14-_jpg.rf.642cb4e31ef48fb6626eeb6b124ff7f5.jpg': [3, 1, 1, 1, 5, 5, 4], 'C-1700-4_600_2mmX-171-_jpg.rf.d8564022058d4e0d7b742ab8c3dba8bf.jpg': [1, 4, 5, 4, 5], '868kg0_2msX-11-_jpg.rf.3ffdc4a65a1aaa54b81dda0173ee47be.jpg': [5, 5, 4, 3], 'cropped_D-1700-4_600_5msX-37-_bmp.rf.e5335ea4029dc3469dea109b23d75ac0.jpg': [4, 2, 2, 3], 'cropped_B-1052-6_40-0_2msX-66-_jpg.rf.97d1eb2d28c882a02abd32f6c5f222f7.jpg': [5, 5, 5, 5, 4, 4, 4, 1, 1, 1, 1, 4, 5], 'D-868-7_140_2msX-79-_jpg.rf.05714597dd0ad9864b99cab4a08558b7.jpg': [3, 3, 3, 3, 3, 3, 3, 3, 2, 4, 4, 4, 5, 1, 1], 'C-1300-5_560_2msX-21-_jpg.rf.1863c90597a42a10ce30e42608f99895.jpg': [4, 4, 3, 2, 3, 5, 2, 2, 1, 1, 1, 2, 2, 2, 4, 2, 4, 4, 3, 3, 1, 1], 'C-1300-5_560_2msX-71-_jpg.rf.2ecf4f74765e9fbd3e650d212752d85e.jpg': [3, 3, 1, 1, 1, 1, 1, 1, 4], 'C-1700-4_600_2mmX-47-_jpg.rf.d64d976bfe5506fa4e8dcd2ab9572db6.jpg': [3, 2, 2, 4, 1, 1, 1, 1], 'cropped_C-1300-5_56-0_2X-107-_png.rf.b2efd9a45969ad9c1ead43243cf4184b.jpg': [1, 5, 1, 4, 5, 4], '868kg0_2msX-18-_jpg.rf.baa2cca2022be58e39dc296c4d0fc425.jpg': [1, 3], 'cropped_C-1700-4_600_2mmX-63-_jpg.rf.b0cafa311078740b1d1538771247a23f.jpg': [2, 2, 2, 2, 2, 5, 3, 3, 4, 4, 1, 1, 3, 3, 1, 1, 1, 1], 'C-1300-5_560_2msX-106-_jpg.rf.d7045c738a3db1550850f879edf31032.jpg': [5, 4, 4, 4, 2, 1, 5, 2], '868kg0_2msX-402-_jpg.rf.97992b91ab0f8b321cc2b2a8773c36c2.jpg': [1, 4, 4, 4, 4], 'cropped_D-868-7_140_2msX-23-_jpg.rf.638f064743fab2313e04acd64219a5e8.jpg': [2, 2, 2, 2, 2, 3, 4, 4, 5, 5, 2, 3, 1, 1, 3, 3, 4, 1, 1, 1, 1, 1], 'C-1300-5_560_2msX-66-_jpg.rf.788c6da07aa1f178978df4955eb214e8.jpg': [4, 4, 2, 4, 2, 1, 1, 1, 1, 1], 'C-868-_7140_2msX-31-_jpg.rf.79b602c74f6484726e917f0998207f39.jpg': [4, 4, 1], 'cropped_868kg0_2msX-256-_jpg.rf.18bf150e715ddfc65a8d1437e1cd65af.jpg': [1, 1, 5, 1, 1, 4, 5, 5, 1, 1, 1], '868kg0_2msX-9-_jpg.rf.844a6e63849953e1f8e47dfbd56bf11e.jpg': [4, 3, 2, 2, 3, 2, 3], 'cropped_B1330-5_56-0_2msX-82-_jpg.rf.8914a50b8529cdda1aea40d49adc2c45.jpg': [4, 4, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'C-868-_7140_2msX-107-_jpg.rf.5527f69c95ad7029626e02a0234feffe.jpg': [3, 4, 2, 2, 4, 4, 5, 3], '-0_2X-51-_jpg.rf.a5b3c04046d275f814620b1af524c7e6.jpg': [3, 1], 'cropped_C-1700-4_600_2mmX-127-_jpg.rf.11b6d02907ed65d28d06998470f6d3be.jpg': [5, 5, 4, 4, 4, 4, 1, 1, 1], 'D-868-7_140_2msX-72-_jpg.rf.cfe9d8848793d0587040f76f28630a92.jpg': [5, 3, 4, 2, 2, 2, 4, 4, 5, 4, 1, 1, 1, 3, 5, 1, 5], 'cropped_B-1700-4_60-0_2X-212-_jpg.rf.e4fc0a64aff063eecb968a22aca7d976.jpg': [1, 1, 4, 4], 'C-868-_7140_2msX-7-_jpg.rf.9eb2fb3a1c25f81816a4bbc7e093c8d4.jpg': [2, 4, 2, 3, 3, 1, 1, 1], 'cropped_C-1300-5_56-0_2X-173-_png.rf.e922f1c30b7ec1124cd53b34bf555fc6.jpg': [5, 4, 4, 4, 2, 2, 2, 3, 3], 'B1330-5_56-0_2msX-92-_jpg.rf.0036da66eb5ecbcaae532bd14bb63da9.jpg': [4, 4, 3, 2, 3, 5, 2, 2, 1, 1, 1, 2, 2, 2, 4, 2, 4, 4, 3, 3, 1, 1], 'C-1300-5_560_2msX-71-_jpg.rf.674c0cb16a761e689176ef82115b89ff.jpg': [4, 4, 4, 4], 'B1330-5_56-0_2msX-9-_jpg.rf.c75a5cf4efd73ebfe8ec59e3738e5f6e.jpg': [4, 4, 4, 4, 4, 2, 2, 2, 3, 3], 'B1330-5_56-0_2msX-57-_jpg.rf.a60b7cb5b69544962f82f5fcfba555aa.jpg': [3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 2, 4, 4, 1, 1], 'C-1300-5_560_2msX-19-_jpg.rf.e2227c60140a6598ccd9f06b993d3d60.jpg': [1, 4, 5, 4, 5], '868kg0_2msX-11-_jpg.rf.04bc8849245c8348725a67be0d585902.jpg': [3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 4, 2, 2, 4], 'C-1300-5_560_2msX-22-_jpg.rf.5b448a8f446f82f5dd5a987190b33f73.jpg': [2, 2, 4, 3, 1], 'cropped_B-1052-6_40-0_2msX-126-_jpg.rf.b03d901e2c8e4da65dff8ab3c2828330.jpg': [3, 2, 2, 4, 4, 5, 3, 1, 1, 1, 1], '-0_2X-168-_jpg.rf.4c9b4c253a45ac34e53aeeb9ca42a89f.jpg': [5, 4, 4, 5, 2, 2, 3, 1, 1, 3, 1], 'cropped_B-1700-4_60-0_2X-375-_jpg.rf.a1b7013971f0ba13995bfbcfe4f825a8.jpg': [4, 2, 2, 2, 2, 2, 5, 1, 5], 'B-1052-6_40-0_2msX-27-_jpg.rf.adcb039a6b8ccbddc668bbc9e33fe170.jpg': [4], 'cropped_868kg0_2msX-465-_jpg.rf.43751e3cadfb06e12b8b920be0d3a962.jpg': [2, 5, 4, 4, 1, 5], '868kg0_2msX-482-_jpg.rf.69e49d5c341b4d9dc264cedc2ade3f95.jpg': [2, 4, 4, 1, 1, 1, 1, 1, 1, 1, 2], 'C-868-_7140_2msX-104-_jpg.rf.c6930c4feb3ace6964d8f1cbd97a25bb.jpg': [5, 4, 4, 4, 4, 2, 2, 4, 4, 3, 3, 3], 'B1330-5_56-0_2msX-89-_jpg.rf.dceb497cc7b6c153d04619d0efc0efc5.jpg': [4, 1], 'cropped_868kg0_2msX-157-_jpg.rf.bbacfd41fa2b2b5f2014c61f1fb94b5a.jpg': [2, 5, 2, 2], 'cropped_B-1052-6_40-0_2msX-39-_jpg.rf.df05e71840caa21036f6ff983b2a90d3.jpg': [5, 4, 4, 4, 4, 4, 1], 'D-868-7_140_2msX-49-_jpg.rf.cd84a9d57006ab2df174de11dcefa43e.jpg': [5, 4, 4, 2, 1, 5], '868kg0_2msX-480-_jpg.rf.af8e25ed2a3b88ed6707107dfcf94e4d.jpg': [4, 4, 5, 5, 5, 1, 4, 3], 'cropped_B-1300-5_56-0_2mmX-157-_jpg.rf.66bb742adc9e24bfbbe8ca1dcaa92220.jpg': [3, 2, 2, 4, 4, 4, 5, 5, 4, 3, 1], '-0_2X-212-_jpg.rf.2d953d50973d80de34c66f6608616df2.jpg': [4, 1], 'cropped_868kg0_2msX-344-_jpg.rf.7d28c859d1a099927a3c9b05a7ece02d.jpg': [4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1], 'C-868-_7140_2msX-64-_jpg.rf.e9198288b52d9370f637175d8b3cb20d.jpg': [5, 5, 4, 2, 3, 1, 1, 1, 1, 1, 3], 'cropped_B-1700-4_60-0_2X-327-_jpg.rf.05dae305784455912a720d9270041378.jpg': [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 3, 4, 5, 5, 1, 3, 1, 3, 5, 2, 3, 1], 'C-1300-5_560_2msX-45-_jpg.rf.cef4eb1253fc6d6596921a741760007a.jpg': [4, 1], 'cropped_868kg0_2msX-439-_jpg.rf.b95f0877d53e0470bd5ae6ea92196f1a.jpg': [4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1], 'C-868-_7140_2msX-67-_jpg.rf.501287988bbc5c9c1178f01f6f27a3e9.jpg': [2, 2, 2, 2, 2, 3, 4, 4, 5, 5, 2, 3, 1, 1, 3, 3, 4, 1, 1, 1, 1, 1], 'C-1300-5_560_2msX-66-_jpg.rf.b45af9d36e1ed651aa568e7888058202.jpg': [4, 4, 4, 2, 2, 3, 3, 2], 'cropped_B1330-5_56-0_2msX-25-_jpg.rf.31ed89c526d4990ddc5acbe460bd828d.jpg': [3, 3, 1, 1, 5, 4, 1], 'C-1700-4_600_2mmX-195-_jpg.rf.1c1b974b53267f25b0d19dbecc7da193.jpg': [3, 3, 3, 2, 1, 1, 1, 4, 4, 3, 1, 1], 'C-1300-5_560_2msX-104-_jpg.rf.4ce8c8f82736332603912a8c4d4c7c84.jpg': [2, 4, 2, 5, 3], 'cropped_B-1052-6_40-0_2msX-88-_jpg.rf.885379c158be697aa4e80dd21030d1ae.jpg': [3, 2, 4, 2, 1, 1, 1, 1, 1], 'cropped_C-1300-5_56-0_2X-84-_png.rf.a8f01805ecf13f6d30758aa1dc0151a1.jpg': [3, 1, 2, 5, 2, 4, 4, 3, 1, 1, 1, 1], 'C-1300-5_560_2msX-103-_jpg.rf.0477b1003309f9df1c03846e16f974cf.jpg': [3, 3, 4, 4, 4, 5, 5, 3, 3, 2, 2, 1, 1, 1, 1], 'C-1300-5_560_2msX-41-_jpg.rf.f26420c457290475704108ba071b27ed.jpg': [1, 5, 4, 4, 4, 4], 'cropped_D-868-7_140_2msX-79-_jpg.rf.fd2e6290136a57d8569bc0235a0abcfc.jpg': [4, 2, 1, 1, 1, 1, 1], 'C-868-_7140_2msX-68-_jpg.rf.5b7afb3ea348d160b841425cd513c245.jpg': [3, 2, 2, 4, 1], 'cropped_C-1300-5_56-0_2X-107-_png.rf.306bd4d9590398261e43bdc41affe502.jpg': [3, 4, 2, 2, 4, 4, 5, 5, 3, 1], '-0_2X-165-_jpg.rf.febf414a50e76c6ff91c15623833d149.jpg': [4, 4, 4, 1, 5, 4, 3], 'cropped_B-1300-5_56-0_2mmX-81-_jpg.rf.dfcae21046b5863f3651e782155079c4.jpg': [4, 4, 4, 4, 5, 4, 1], 'D-868-7_140_2msX-10-_jpg.rf.f0149593a4167c9811e708f24912a254.jpg': [1, 1, 1, 2, 3, 3, 5, 2, 2, 2, 2, 4, 2, 4, 1, 3, 1, 1], 'C-1300-5_560_2msX-70-_jpg.rf.3c3ee10a7e17bdf85df84cbc0d441e9a.jpg': [1, 1, 1, 1, 1, 1, 1, 1, 4], 'C-868-_7140_2msX-9-_jpg.rf.3a5574cc5438f43b5fe4ba64f0b021ee.jpg': [5, 4, 1, 1, 5, 5, 2], '868kg0_2msX-130-_jpg.rf.788d3786bf72688dfd97cff3f2ee99d3.jpg': [5, 4, 4, 4, 4, 4, 4, 1, 4, 5, 5], 'D-868-7_140_2msX-107-_jpg.rf.ddf66e539f65e0daef1999745269b326.jpg': [2, 4, 5, 2, 2, 2, 3, 5, 1, 1, 1, 1, 1], 'B-1052-6_40-0_2msX-174-_jpg.rf.7c6c91aed5299ecdc7d08befb9678f90.jpg': [4, 1], 'cropped_C-868-_7140_2msX-59-_jpg.rf.10bd97cd2e814c355420726bcf80100d.jpg': [4, 5, 1], 'cropped_868kg0_2msX-25-_jpg.rf.0d91c440f2ea22246f39ed6eb46afb09.jpg': [3, 3, 3, 3, 3, 3, 3, 3, 2, 4, 4, 4, 5, 1, 1], 'C-1300-5_560_2msX-21-_jpg.rf.b2025b42e2c9819e7d18c0cefb85a958.jpg': [4, 4, 4, 4, 4, 2, 2, 2, 3, 3, 3], 'B1330-5_56-0_2msX-56-_jpg.rf.8eae3bfe64f1fa7f491c2692a83448de.jpg': [4, 1], 'cropped_C-868-_7140_2msX-22-_jpg.rf.12b520c4ce4b85e3f91067b82ad4c528.jpg': [5, 4, 4, 4, 4, 4, 3, 4, 5, 4], 'D-1700-4_600_5msX-12-_jpg.rf.a491e41fd35d718bf8ee0eca21c04315.jpg': [2, 5, 4, 4, 1, 5], '868kg0_2msX-482-_jpg.rf.427a2fdc24e9d807242d6d6feeee86b0.jpg': [3, 3, 3, 4, 4, 4, 4, 5, 5, 2, 4, 2, 1, 1], 'C-1300-5_560_2msX-26-_jpg.rf.0771b19e3cf5b01ec5cf3792ba8a319b.jpg': [4, 4, 4, 4, 4, 5, 1], 'C-1700-4_600_2mmX-27-_jpg.rf.079a0184e6660f3d69271b1f5a0764c7.jpg': [5, 4, 2, 1, 5], '868kg0_2msX-481-_jpg.rf.7513697d5e99846abc09027370b8dc16.jpg': [3, 1], 'cropped_C-1700-4_600_2mmX-186-_jpg.rf.7e17c938540db79ba2ab0968a3ed44f5.jpg': [5, 5, 4, 3, 3], 'D-1700-4_600_5msX-10-_jpg.rf.fec2ad853816798cc8d4b2067ab6d898.jpg': [4, 1, 1], 'cropped_C-868-_7140_2msX-35-_jpg.rf.9fe2f914d49be825fca1602214a5ce2c.jpg': [2, 5, 2, 2], 'cropped_B-1052-6_40-0_2msX-39-_jpg.rf.f8f893fb9d756376033b5ac0ce8e367a.jpg': [3, 2, 2, 4, 4, 5, 3, 1, 1, 1, 1], '-0_2X-168-_jpg.rf.8bff265c8a361ca970d5292ead14a5f5.jpg': [4, 4, 5, 4, 4, 1, 1, 4, 4, 1, 4, 4, 4, 4], 'D-868-7_140_2msX-40-_jpg.rf.d1e5680cd1ea8b775809d1eeaaec07d1.jpg': [5, 4, 2, 4, 3, 1, 1, 1, 3, 1], 'cropped_B-1700-4_60-0_2X-374-_jpg.rf.28e930587ffaa5fd534efd8b2a74792c.jpg': [3, 3, 3, 3, 3, 3, 3, 3, 2, 4, 4, 4, 5, 1, 1], 'C-1300-5_560_2msX-21-_jpg.rf.927c258485a259b2a8c02f309ce07dde.jpg': [5, 5, 5, 5, 4, 4, 4, 1, 1, 4, 5], 'D-868-7_140_2msX-105-_jpg.rf.0268baa4c330144cdaf326b7dfd730c7.jpg': [5, 4, 4, 5, 2, 2, 3, 1, 1, 3, 1], 'cropped_B-1700-4_60-0_2X-375-_jpg.rf.e9fe1dba6ecf2ca69b2009211eb1cb50.jpg': [3, 1, 1, 1], 'cropped_C-1700-4_600_2mmX-197-_jpg.rf.574418a131873cc2396c8c5fdd8fafca.jpg': [1, 1, 2, 5, 3, 2, 5, 2, 2, 2, 4, 2, 4, 4, 4, 1, 3, 1, 1, 1, 3, 3], 'C-1300-5_560_2msX-63-_jpg.rf.8981f81ac2289da5ce0e1c9966826379.jpg': [4, 2, 2, 3], 'cropped_B-1052-6_40-0_2msX-66-_jpg.rf.8710dab2090df15541720bf8f96b9901.jpg': [5, 5, 5, 5, 5, 4, 4, 4, 4, 1, 1, 1], 'D-868-7_140_2msX-70-_jpg.rf.c1eb5a6caee62c1b13198e536d7f7cb8.jpg': [4, 2, 1, 1], 'C-868-_7140_2msX-65-_jpg.rf.a3451ed823d9cc3c60a997acda8acd82.jpg': [1, 1, 5, 1, 1, 4, 5, 5, 1, 1, 1], '868kg0_2msX-9-_jpg.rf.0a5d0ce45de872493ce8cdbc2767a8ce.jpg': [5, 5, 4, 3, 3], 'D-1700-4_600_5msX-10-_jpg.rf.bacd079411042b41710c0998988a4d9d.jpg': [4, 4, 2, 2, 2, 2], 'cropped_B1330-5_56-0_2msX-13-_jpg.rf.05530600fcfbe8786e8ba5cf7d340787.jpg': [4, 1], 'cropped_868kg0_2msX-157-_jpg.rf.af47ce47b05130f61b10d296ffc93efa.jpg': [3, 4, 2, 2, 2, 4, 4, 5, 5, 5, 3, 5, 1, 1, 1, 1, 1], '-0_2X-164-_jpg.rf.c61e2ace93e530be113654be368bc492.jpg': [4, 2, 1, 1, 1, 1, 1, 1, 1], 'C-868-_7140_2msX-63-_jpg.rf.df89b043420cab1031b9377b9c7b6539.jpg': [3, 4, 4], 'cropped_C-1300-5_56-0_2X-35-_png.rf.8d5d508edf3df9809dfb2a9519b712c9.jpg': [5, 4, 4, 4, 4, 1, 1, 5, 1], '868kg0_2msX-17-_jpg.rf.72aed590a98ae3fbf05dcf38846344ac.jpg': [1, 1, 4, 4], 'C-868-_7140_2msX-7-_jpg.rf.1afc3079a627d7d02f3e2bf91051b0e8.jpg': [1, 5, 4, 4, 4], 'cropped_D-868-7_140_2msX-69-_jpg.rf.443427a582ef5dda0fa5be19916afe66.jpg': [3, 4, 4, 2, 2, 1, 1, 3, 5, 5, 5, 5], 'cropped_B-1700-4_60-0_2X-186-_jpg.rf.697e12f467e1de3461bbed521128ab7d.jpg': [5, 3, 4, 2, 2, 2, 4, 4, 5, 4, 1, 1, 3, 5, 1, 5], 'cropped_B-1700-4_60-0_2X-212-_jpg.rf.1fdd23f9daad7e5e1ec0c7a56f8afb30.jpg': [5, 5, 5, 5, 4, 4, 4, 1, 1, 1, 1, 4, 5], 'D-868-7_140_2msX-79-_jpg.rf.2f9dee64af433ae8de719a52adbc66f9.jpg': [5, 5, 5, 5, 4, 4, 4, 1, 1, 1, 1, 4, 5], 'D-868-7_140_2msX-79-_jpg.rf.247ec8074dfdc0c1cdcdb1f14929ed89.jpg': [3, 2, 2, 4, 1, 1, 1, 1], 'cropped_C-1300-5_56-0_2X-107-_png.rf.54e9e4e77238ed1d377baf9cb16d41e7.jpg': [4, 4, 4, 4, 4, 1, 5, 3], 'cropped_B-1300-5_56-0_2mmX-92-_jpg.rf.81f376a1ac08c73eb395d77821c205be.jpg': [5, 4, 4, 4, 4, 1, 1, 5, 1], '868kg0_2msX-17-_jpg.rf.5b737749db098a06cf3f47d256e5f0cb.jpg': [3, 4, 2, 2, 2, 4, 4, 5, 5, 5, 3, 5, 1, 1, 1, 1, 1], '-0_2X-164-_jpg.rf.1a3f2a406680d85c62d90e0c7c91bfe5.jpg': [3, 4, 2, 2, 4, 4, 5, 5, 5, 3, 1, 1], '-0_2X-163-_jpg.rf.7483ce536cf807743eb4a23ea3d5cd24.jpg': [3, 4, 2, 2, 4, 4, 5, 3], '-0_2X-51-_jpg.rf.43747b5a7bf238ae503aaa3d3e784716.jpg': [3, 4, 2, 2, 4, 4, 5, 5, 5, 5, 3, 1], '-0_2X-200-_jpg.rf.7120d2b9ef37930b52caa1e804b6d74b.jpg': [4, 4, 2, 2, 2, 2], 'cropped_B1330-5_56-0_2msX-13-_jpg.rf.f724a2e2f7db3525be7df969c6acc3ee.jpg': [4, 2, 2, 3], 'cropped_B-1052-6_40-0_2msX-66-_jpg.rf.9ae34e85121f93bafad0f75ed4eabdd4.jpg': [4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1], 'C-868-_7140_2msX-64-_jpg.rf.eb2963eb116827b719ed94db6f20bbc0.jpg': [1, 1, 5, 1, 1, 4, 5, 5, 1, 1, 1], '868kg0_2msX-9-_jpg.rf.5debce39544e8aac09c6998213215b1c.jpg': [5, 4, 4, 4, 2, 2, 2, 3, 3], 'B1330-5_56-0_2msX-92-_jpg.rf.ec1403b8027c46c7c8b909c5fb65332e.jpg': [5, 4, 3, 3, 5, 5, 4], 'D-1700-4_600_5msX-14-_jpg.rf.6fb034e46a5d04368f58fb67f1339659.jpg': [1, 1, 2, 5, 3, 2, 5, 2, 2, 2, 4, 2, 4, 4, 4, 1, 3, 1, 1, 1, 3, 3], 'C-1300-5_560_2msX-63-_jpg.rf.3a9c77b986e47a80fb523eb8ed7eaad7.jpg': [4, 4, 5, 5, 2, 4, 1, 1, 3], 'cropped_B-1300-5_56-0_2mmX-180-_jpg.rf.2db5682ccaf3b7894c0dd260185f4c97.jpg': [5, 5, 5, 5, 4, 4, 4, 1, 1, 4, 5], 'D-868-7_140_2msX-105-_jpg.rf.979a1c3d56d39e2b0895a0a74e689f56.jpg': [3, 1], 'cropped_C-1700-4_600_2mmX-127-_jpg.rf.544df84d4d661f39240cce2fbf0e4e2f.jpg': [3, 5, 4, 3, 1, 1, 1, 1], 'C-1700-4_600_2mmX-196-_jpg.rf.10af538d679cd1b1c14929ae4140c1b6.jpg': [5, 4, 5, 5, 1, 2, 2], '868kg0_2msX-128-_jpg.rf.a94b6f88201fcdf3a44c1474db79c4be.jpg': [5, 5, 5, 5, 5, 4, 4, 4, 4, 1, 1, 1], 'D-868-7_140_2msX-70-_jpg.rf.84a6c5a277d113ff517a8aeb72be727f.jpg': [1, 4, 4, 4, 5], 'D-868-7_140_2msX-26-_jpg.rf.bd20922d53e9c1990f49c2400ec19a45.jpg': [3, 3, 4], 'D-1700-4_600_5msX-9-_jpg.rf.d4da42e70304fefd1f3ded5316698841.jpg': [3, 4, 2, 2, 1, 1, 1], 'cropped_C-1300-5_56-0_2X-148-_png.rf.3de8551f32bdde6040568fbf3973acfe.jpg': [3, 5, 3, 1, 4], 'C-1700-4_600_2mmX-197-_jpg.rf.e0cefe38b0145d5f1755c74322b1b702.jpg': [1, 5, 5, 4, 4, 4, 1, 1], 'cropped_D-868-7_140_2msX-200-_jpg.rf.7fe47926eb847658a8417e03c7c03d8b.jpg': [3, 4, 2, 2, 4, 4, 5, 3], '-0_2X-51-_jpg.rf.c8a4019ced71b710f313c27b813b639e.jpg': [5, 4, 4, 4, 4, 1, 4], '-0_2X-18-_jpg.rf.3d2d782d5834b7591e29a022eec2fe2b.jpg': [3, 1], 'cropped_C-1700-4_600_2mmX-186-_jpg.rf.274268c94d1430fc72df09513a57af14.jpg': [4, 1], 'cropped_868kg0_2msX-439-_jpg.rf.ea62bd4ad3743d098ead49b4fb6fb681.jpg': [4, 2, 2, 2, 2, 2, 2, 1, 1, 5], 'B-1052-6_40-0_2msX-30-_jpg.rf.7509d14e91b36aca276340bb2c3ba717.jpg': [4, 2, 2, 4, 1], 'cropped_C-868-_7140_2msX-106-_jpg.rf.65dd1335ae96e17c5524eb44adab816c.jpg': [3, 4, 4, 2, 2, 4, 4, 5, 5, 5, 5, 5, 5, 3, 1, 1], '-0_2X-209-_jpg.rf.38572602a14a8590713a20d844fe3ac3.jpg': [3, 1, 4], 'cropped_C-1700-4_600_2mmX-51-_jpg.rf.a6c57df2017507f64864466a2c54d0eb.jpg': [1], 'cropped_868kg0_2msX-2-_jpg.rf.ac861d27c898bac90f0ef72e6660b916.jpg': [3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 2, 4, 4, 1, 1], 'C-1300-5_560_2msX-19-_jpg.rf.deb33ca31ce1e819fe71bffc9962aa66.jpg': [3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 2, 4, 4, 1, 1], 'C-1300-5_560_2msX-19-_jpg.rf.5caf06b8b532b12729903c567b309a01.jpg': [4, 4, 4, 4, 4, 2, 2, 2, 3, 3], 'B1330-5_56-0_2msX-57-_jpg.rf.b9f83847340717fc9d30887320f603ef.jpg': [4, 4, 4, 4, 4, 2, 2, 2, 3, 3], 'B1330-5_56-0_2msX-57-_jpg.rf.f48a76d4d1dc6e370e73402b6fe5c190.jpg': [5, 3, 4, 2, 2, 2, 4, 4, 5, 4, 1, 1, 3, 5, 1, 5], 'cropped_B-1700-4_60-0_2X-212-_jpg.rf.53d48564ef44242565da70d50f270434.jpg': [3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 4, 2, 2, 4], 'C-1300-5_560_2msX-22-_jpg.rf.c87814414486161a0a4da8cfe4738abd.jpg': [5, 4, 4, 4, 1, 5], '868kg0_2msX-8-_jpg.rf.159618b624512c1475ba523170419451.jpg': [4, 4, 2, 1, 1, 1, 1, 2, 5, 1, 1], 'C-868-_7140_2msX-145-_jpg.rf.b2beb5a221c07b1b6f9763dec5a56594.jpg': [4, 1, 1], 'cropped_868kg0_2msX-190-_jpg.rf.00fa372ddb8b56e753e7a5f8dcd0de33.jpg': [4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'C-868-_7140_2msX-147-_jpg.rf.e3685b504a9e09eba78c6e1d658c8d38.jpg': [4, 4, 4, 4, 4, 4, 4, 5, 4, 1], 'D-868-7_140_2msX-11-_jpg.rf.80d054d81c97ac4b60f81f68b28003d8.jpg': [1], 'cropped_D-868-7_140_2msX-0-_jpg.rf.123a9069bc5511e9995c1bbdbaff045c.jpg': [4, 5, 3, 2, 5, 5, 5, 1, 3], 'cropped_B-1700-4_60-0_2X-340-_jpg.rf.b4010b38e961c5935d3429ca3d54e415.jpg': [3, 4, 4, 2, 2, 2, 2, 2, 2], 'cropped_C-1300-5_56-0_2X-47-_png.rf.192dc99180ed132b88326861c267af01.jpg': [3, 2, 4, 1, 1, 1, 1], 'cropped_C-1300-5_56-0_2X-114-_png.rf.6a41c1d9e296404864917e43bde30e6b.jpg': [4, 5, 4, 3, 4, 2, 2, 3, 1, 5, 5, 5], '-0_2X-210-_jpg.rf.cb4a0567f7a86d90c96dbb64fbb4d6ec.jpg': [3, 4, 2, 2, 4, 4, 4, 5, 5, 3, 5, 5, 1, 1], '-0_2X-211-_jpg.rf.086523d3aa439469619df7b4abc0bcb4.jpg': [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 3, 4, 5, 5, 1, 3, 1, 3, 5, 2, 3, 1], 'C-1300-5_560_2msX-45-_jpg.rf.f58f33d0766c0b2c5a6383fe9f23ef09.jpg': [5, 4, 4, 5, 4, 1, 1, 5], 'D-868-7_140_2msX-108-_jpg.rf.3b49c6100ca83692f083671f11b19b69.jpg': [1, 4, 1], 'cropped_C-868-_7140_2msX-80-_jpg.rf.187459fd893ada30259f80c508bd3a82.jpg': [4, 1], 'cropped_C-868-_7140_2msX-22-_jpg.rf.f6def9c637e481d24c8bfca4b76086ab.jpg': [3, 3, 3, 4], 'D-1700-4_600_5msX-8-_jpg.rf.891aaaedaabc2dba6688b3d8b76bd096.jpg': [1, 1, 4, 2, 2, 2, 2, 3], 'cropped_C-1300-5_56-0_2X-60-_png.rf.78d6cdfd03f7f606bbf077fcaddb90ec.jpg': [4], '-0_2X-17-_jpg.rf.147138709fbdc955bb063be5afe9d550.jpg': [3, 1, 4], 'cropped_C-1700-4_600_2mmX-51-_jpg.rf.3668e79c4b58a97be7ee9a51e6bf0a94.jpg': [1, 4, 1, 3, 3], 'C-1700-4_600_2mmX-71-_jpg.rf.c34d90e58fa2285040857ad26bd48788.jpg': [5, 4, 4, 4, 4, 4, 1, 1, 5], 'D-868-7_140_2msX-39-_jpg.rf.7d2eec9087d53f284dedbf1a30dc6125.jpg': [4, 4, 4, 4, 1, 5], 'D-868-7_140_2msX-13-_jpg.rf.f0d507812b601b813c8471bd148e0e26.jpg': [4, 4, 4, 4, 2, 2, 4, 3, 3, 3], 'B1330-5_56-0_2msX-90-_jpg.rf.780af2cc0dd0c771d5e4c752a96c1923.jpg': [5, 1, 5, 4, 4], '868kg0_2msX-10-_jpg.rf.c8855244fc6368f339cdd218fd3e1f0d.jpg': [2, 2, 2, 1, 1, 4, 1, 1, 1, 3, 3, 3, 3, 5, 2, 5], 'C-1300-5_560_2msX-73-_jpg.rf.b4f58f46657aef2553d24d842bf72c17.jpg': [3, 2, 1, 1, 1, 4, 1, 3, 5, 5], 'C-1300-5_560_2msX-134-_jpg.rf.5af8e21a8567038a96621a506b89595f.jpg': [5, 4, 3, 3, 2, 2, 5, 5, 1, 1, 1, 5], 'B-1052-6_40-0_2msX-170-_jpg.rf.c728e354c3235ebc23f2e61b18c6329b.jpg': [4, 4, 4, 4, 4, 2, 2, 2, 3, 3, 3, 3, 3, 2], 'B1330-5_56-0_2msX-55-_jpg.rf.3eda27ce069b59584e7b2f37eda53441.jpg': [1, 1, 4, 4, 4, 5], 'D-868-7_140_2msX-23-_jpg.rf.a84e444616f971cb39f8b366bb8d1eff.jpg': [4, 2, 2, 2, 1], 'cropped_B-1052-6_40-0_2msX-27-_jpg.rf.f162d5f2ffb202c1b4db6918824f4b34.jpg': [4, 2, 2, 2, 2, 2, 2, 2, 5, 1], 'B-1052-6_40-0_2msX-31-_jpg.rf.6051d2f85fb7f60244e7efd3eee34f6b.jpg': [5, 3, 4, 2, 1, 1, 1, 3], 'cropped_B-1700-4_60-0_2X-92-_jpg.rf.37a947e9cc629911be5abaea79b9feb8.jpg': [4, 2, 2, 2, 2, 3], 'cropped_B-1052-6_40-0_2msX-51-_jpg.rf.97e78860a34ac2c8450c8d78c7f07ab1.jpg': [1, 1, 1, 5, 5, 4, 4, 4], 'cropped_D-868-7_140_2msX-133-_jpg.rf.588df94c02273bf85286d258fce60cea.jpg': [4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1], 'C-868-_7140_2msX-67-_jpg.rf.990e8795109877738ebdabe1bfd5d40c.jpg': [5, 5, 4, 5, 4, 3, 3, 1], 'D-1700-4_600_5msX-48-_jpg.rf.6ac2834f7446fb101cf954893b05be37.jpg': [4, 1, 1, 2, 2], 'C-868-_7140_2msX-34-_jpg.rf.a7b9066665e812926b783da024542592.jpg': [4, 4, 4, 4], 'C-1700-4_600_2mmX-18-_jpg.rf.8463171e56e6668beb93f825f5af3c51.jpg': [3, 4, 4, 4, 2, 4, 4, 5, 5, 5, 1, 1, 4, 3, 5], 'C-1300-5_560_2msX-23-_jpg.rf.60ad495f791ed53801bba654763304df.jpg': [2, 5, 3, 2, 2, 2, 2, 2, 2, 4, 4, 1, 3, 3, 1, 1, 1, 1], 'C-1300-5_560_2msX-64-_jpg.rf.0497784624a2cdafd6285691a513cd10.jpg': [1, 1, 1, 5, 4, 4, 5, 4, 5, 5, 1, 1, 1], 'D-868-7_140_2msX-69-_jpg.rf.ff405ca53a616e6cf45a86da5202cba5.jpg': [5, 4, 4, 4, 4, 1, 1, 1, 1, 1, 5, 5, 4, 4], 'D-868-7_140_2msX-68-_jpg.rf.c0c14b47df0f541c4e0f119b4a33e6b5.jpg': [4, 1, 1], 'cropped_C-868-_7140_2msX-35-_jpg.rf.1d756f654972e5d698475e04bc627759.jpg': [3, 1, 4], 'cropped_C-1700-4_600_2mmX-44-_jpg.rf.4d36804e66e2dbde2ccb8ff275440d89.jpg': [3, 2, 3, 3, 3, 3, 3, 2, 2, 4, 1, 1, 1, 1, 4, 1, 1, 1, 5, 5], 'C-1300-5_560_2msX-108-_jpg.rf.57d74eeb719536ab86b9a2ee27d749f4.jpg': [3, 4, 2, 2, 4, 4, 5, 5, 3, 1], '-0_2X-165-_jpg.rf.61df794436b8835291913e53ef13fead.jpg': [5, 5, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 5, 5], 'D-868-7_140_2msX-196-_jpg.rf.632ef48393fa7e847fadc47206dca138.jpg': [3, 4, 5, 2, 4, 2, 3, 1], 'cropped_B-1700-4_60-0_2X-87-_jpg.rf.43e8f33dbffe9657c0cf44a68fe6c273.jpg': [4, 4, 4, 5, 5, 3, 2, 4, 4, 1, 1, 1], 'C-1300-5_560_2msX-27-_jpg.rf.96a10b4eb5dcfedb61fda8a4774b548c.jpg': [3, 5, 3, 4, 4, 5, 5, 5, 1, 1, 2], 'cropped_B-1700-4_60-0_2X-287-_jpg.rf.02f6d2e3d0508970542cd179fe422479.jpg': [5, 4, 4, 4, 4, 1, 5, 5, 4], 'D-868-7_140_2msX-52-_jpg.rf.62050ed209a7df2ec75daaa60a55c000.jpg': [4, 4, 1, 1], 'cropped_C-868-_7140_2msX-113-_jpg.rf.b0f491bc2b2762aab8c6dc6e8b29ed05.jpg': [4, 2, 2, 2, 2, 1, 5], 'B-1052-6_40-0_2msX-12-_jpg.rf.66ee0efaff5a1897a8300c441aec8f19.jpg': [4, 2, 2, 2, 2, 2, 1, 5], 'B-1052-6_40-0_2msX-29-_jpg.rf.e8f73ee3c01a9c1192eb18653f92cc77.jpg': [1, 4, 4, 4, 4], 'C-868-_7140_2msX-8-_jpg.rf.1023d8875ff7a4a696547b57f9dd43ad.jpg': [4, 4, 5, 5, 2, 4, 1, 1, 3], 'cropped_B-1300-5_56-0_2mmX-180-_jpg.rf.0d53187094a2ddae20bfbbfc46db3e08.jpg': [2, 2, 2, 1, 1, 1, 5, 5, 4, 4, 3, 3, 5, 1, 1, 1, 1, 4, 3], 'C-1300-5_560_2msX-109-_jpg.rf.624cf7e26955b149eb39f733248fb147.jpg': [1, 2, 4, 5, 5], '868kg0_2msX-404-_jpg.rf.11fe5f8aae3cef32fde1a0d9095d8c68.jpg': [2, 4, 5, 2, 2, 2, 3, 5, 1, 1, 1, 1, 1], 'B-1052-6_40-0_2msX-174-_jpg.rf.02e98d5d90a33a76bb7251819e822e0b.jpg': [3, 2, 2, 2, 2, 4, 4, 5], 'cropped_B-1700-4_60-0_2X-40-_jpg.rf.3f8a0ef39e9eafb3d75c2f509a5e66a1.jpg': [5, 4, 4, 4, 4, 4, 1], 'D-868-7_140_2msX-49-_jpg.rf.e01b80ec5ccef8652c45271f7cda948e.jpg': [3, 3, 4, 2, 4, 4, 5, 5, 5, 5, 3, 1], '-0_2X-375-_jpg.rf.31ffdd93fd826d946233120e488ea7f9.jpg': [3, 1, 1, 2, 4, 1, 4], 'cropped_C-1300-5_56-0_2X-171-_png.rf.875b681175ab4a6201ef61349f727894.jpg': [1, 4, 5, 5, 5, 2, 2, 2, 2, 3, 3, 2, 1], 'B-1052-6_40-0_2msX-82-_jpg.rf.2dcadf5692c66550a9580ff7f4c56f8d.jpg': [4, 4, 2, 2, 2, 3, 3, 3], 'cropped_B1330-5_56-0_2msX-48-_jpg.rf.c2639c75036c556c28b546f35a9a3574.jpg': [4, 2, 2, 5, 4], 'B1330-5_56-0_2msX-7-_jpg.rf.11c2850043028cc9d743f718aa91d629.jpg': [4, 3, 4, 2, 3, 2, 2], 'cropped_B1330-5_56-0_2msX-71-_jpg.rf.3f1ab65fe5c449778b050e29fb6cfebe.jpg': [4, 1], 'cropped_C-868-_7140_2msX-59-_jpg.rf.e94d309872595a0097ad2c63a17d1a43.jpg': [4, 2, 1, 1, 1, 1, 1], 'C-868-_7140_2msX-68-_jpg.rf.1be26d4a5b6aeca3c109b3e091c42cd7.jpg': [4, 1, 2, 5, 2], 'B-1052-6_40-0_2msX-9-_jpg.rf.b49ed3d7187d16d77211c425af8f9771.jpg': [4, 5, 1], 'B-1052-6_40-0_2msX-8-_jpg.rf.21b72884a6ee2a86a41b139d81f90e51.jpg': [5, 4, 4, 2, 5, 1, 1], '868kg0_2msX-129-_jpg.rf.da5ec38eeded976c46067592c53fe2cc.jpg': [4, 4, 4, 1, 5, 4, 3], 'cropped_B-1300-5_56-0_2mmX-81-_jpg.rf.fb26afdfbbf5d35ef2d7bd0542e79476.jpg': [3, 4, 2, 2, 4, 5, 4, 3, 1, 1, 1], '-0_2X-64-_jpg.rf.8d4faf37526a6fe4a187fa8dd9ed4261.jpg': [3, 3, 1, 1, 1, 1, 1, 1, 4], 'C-1700-4_600_2mmX-47-_jpg.rf.77958ca94b98d5532bacb85a63e19787.jpg': [4, 4, 2, 1, 1, 1, 2, 1], 'C-868-_7140_2msX-148-_jpg.rf.38058fab2c8b2bbc61f89bc878bb76a2.jpg': [4, 4, 1, 4], 'cropped_C-868-_7140_2msX-135-_jpg.rf.74598472f4b4abc55fad5ae09265187f.jpg': [5, 3, 4, 2, 2, 5, 5, 5, 3, 1], 'cropped_B-1700-4_60-0_2X-161-_jpg.rf.94a77529a02dc10d2f847bcddd5fc6aa.jpg': [5, 4, 4, 1, 5, 2], '868kg0_2msX-405-_jpg.rf.ff17a1d3c89765312785793bc30ca95e.jpg': [1, 5, 4, 5, 2], '868kg0_2msX-403-_jpg.rf.0131d2504e956288a264e505348f8e73.jpg': [1, 1, 1, 4, 3, 3], 'C-1700-4_600_2mmX-74-_jpg.rf.9fe4631b9f2b6ccb564d2751a0800d74.jpg': [5, 5, 5, 5, 4, 5, 4, 4, 4, 1, 1], 'D-868-7_140_2msX-76-_jpg.rf.521c6bda3e622f9aa81c3533150fdc76.jpg': [3, 5, 3, 1, 4], 'C-1700-4_600_2mmX-197-_jpg.rf.ec53ba254c43a19dbb2ba1c30e205f04.jpg': [4, 2, 2, 1, 1, 1, 1], 'C-868-_7140_2msX-66-_jpg.rf.ff3883cf39bfa49d3ee3e7da01f443a9.jpg': [1, 4, 5, 5, 5, 2, 2, 2, 2, 3, 3, 2, 1], 'B-1052-6_40-0_2msX-82-_jpg.rf.4923f32df21285830eac4941e8e1a00e.jpg': [1, 3, 5, 2, 2, 2, 2, 2, 4, 4, 1, 5, 1, 1, 2, 2, 3, 3, 3, 3, 3, 1, 1, 1], 'C-1300-5_560_2msX-72-_jpg.rf.e2b73fbd373c73e0cd5481eff323e2d5.jpg': [3, 4, 2, 2, 4, 5, 4, 3, 1, 1, 1], '-0_2X-64-_jpg.rf.69b2441be403cc7f7fce5fe824f00015.jpg': [5, 4, 1, 1, 5, 5, 2], '868kg0_2msX-130-_jpg.rf.40e605387b4fa8e0775a25f575c04efb.jpg': [3, 3, 3, 2, 1, 1, 1, 4, 4, 3, 1, 1], 'C-1300-5_560_2msX-104-_jpg.rf.60adc549f4e5957ef078190275083c13.jpg': [5, 4, 4, 4, 5, 5, 5, 5, 1, 1], '868kg0_2msX-16-_jpg.rf.0f25582b46ae86b2a4c765bb474a2d23.jpg': [1, 2, 4, 5, 5], '868kg0_2msX-404-_jpg.rf.f1cfe8c084d4a6c2f8e3e740e81a24de.jpg': [3, 1], 'cropped_C-1700-4_600_2mmX-160-_jpg.rf.e05055f86b7738a1e4dc0644f5a3960d.jpg': [5, 4, 2, 1, 5], '868kg0_2msX-481-_jpg.rf.ce44f0726362623b0b492dede475ab52.jpg': [2, 2, 2, 2, 2, 5, 3, 3, 4, 4, 1, 1, 3, 3, 1, 1, 1, 1], 'C-1300-5_560_2msX-106-_jpg.rf.f8e97cdb7a87a4a356b7c4468617c438.jpg': [1, 4, 1, 3, 3], 'C-1700-4_600_2mmX-71-_jpg.rf.447a7816e9747edd426bbd4c0c8e8917.jpg': [4, 2, 2, 5, 4], 'B1330-5_56-0_2msX-7-_jpg.rf.025de6296ce190c2cd04ba3271b23ad7.jpg': [3, 1, 1, 4, 3, 3, 2], 'C-1300-5_560_2msX-169-_jpg.rf.eac10091f4986c6908faf8e033eeb823.jpg': [4, 2, 2, 1, 1, 1, 1], 'C-868-_7140_2msX-66-_jpg.rf.b3fc7ca615245ad9fe3162b81212308c.jpg': [2, 4, 2, 4, 4, 5, 5, 3], '-0_2X-16-_jpg.rf.735bff904395fc9a0e21bd5762f1c79e.jpg': [4, 4, 2, 2, 2, 3, 3], 'cropped_B1330-5_56-0_2msX-92-_jpg.rf.f865349596364c7ca70dc409ced82302.jpg': [3, 3, 3, 5, 4], 'cropped_D-1700-4_600_5msX-12-_bmp.rf.2b6a11830a2ab803b740a1d3d6fc1f0a.jpg': [4, 2, 2, 1, 2], 'B-1052-6_40-0_2msX-11-_jpg.rf.ed47d7059ed001360df692c419672424.jpg': [4, 2, 1, 1], 'C-868-_7140_2msX-65-_jpg.rf.22f0b59c97148fa0bba693c3250b6aa5.jpg': [5, 4, 4, 4, 4, 1, 4], 'D-868-7_140_2msX-78-_jpg.rf.0a31dbeeaa7364f59ea8064a56e9de67.jpg': [3, 3, 4, 2, 2, 4, 4, 5, 5, 5, 3, 1], '-0_2X-167-_jpg.rf.c66d513b19aa82928d6c2100f37ffb90.jpg': [4, 5, 3, 2, 5, 5, 5, 1, 3], 'cropped_B-1700-4_60-0_2X-340-_jpg.rf.d2393eefb45646dbdd3bedfab7d111de.jpg': [1, 5, 4, 4, 4, 4], 'cropped_D-868-7_140_2msX-79-_jpg.rf.70fc834ee027282676dbcc2e28ae858a.jpg': [4, 2, 2, 4, 1], 'cropped_C-868-_7140_2msX-100-_jpg.rf.fa398879b87675e71125f01698ee4d1a.jpg': [3, 2, 2, 2, 2, 4, 1, 1, 1], 'cropped_C-1300-5_56-0_2X-72-_png.rf.82a6d18e75432002a79b2eb2e872b685.jpg': [2, 2, 2, 3, 1, 1, 4, 1, 1, 1, 3, 4, 5, 1], 'C-1300-5_560_2msX-74-_jpg.rf.bc5a1c82ee91ffb706fcd5b21065139e.jpg': [2, 5, 3, 2, 2, 2, 2, 2, 2, 4, 4, 1, 3, 3, 1, 1, 1, 1], 'C-1300-5_560_2msX-64-_jpg.rf.f54e407437a74b8d8544031f12461b58.jpg': [4, 4, 2, 2, 2, 3, 3], 'cropped_B1330-5_56-0_2msX-92-_jpg.rf.ddc7feda9128adb899a207b43c35140c.jpg': [3, 4, 2, 2, 4, 4, 5, 5, 5, 5, 5, 3, 1], '-0_2X-197-_jpg.rf.2171f68ff6ef81012672745e6d7e6b8c.jpg': [3, 1, 1, 1, 5, 2, 3, 5, 2, 2, 2, 2, 2, 4, 4, 4, 5, 1, 1, 1], 'C-1300-5_560_2msX-47-_jpg.rf.cd6627bbf308361001cb011541422890.jpg': [1, 4, 5, 4, 4, 5], 'cropped_D-868-7_140_2msX-34-_jpg.rf.3993fdcebbedeca2193b6146bcb9c48e.jpg': [3, 2, 2, 5, 4, 3, 3, 3, 1, 1, 1, 1, 5, 5, 5, 5], 'cropped_B-1700-4_60-0_2X-315-_jpg.rf.ed0ed94b3358d3422530128c25de6bcc.jpg': [4, 2, 2, 2, 2, 2, 2, 1, 1, 5], 'B-1052-6_40-0_2msX-30-_jpg.rf.de37c8e04c4cd7e312206213b9f1f3f4.jpg': [5, 5, 4, 3], 'cropped_D-1700-4_600_5msX-37-_bmp.rf.b208dc552052bbf6e6539134f63a77b2.jpg': [4, 4, 1, 1, 4, 1, 1], 'cropped_C-868-_7140_2msX-102-_jpg.rf.5a3999d5643abdf34ebb056646c85365.jpg': [4, 4, 1], 'cropped_868kg0_2msX-256-_jpg.rf.55d25b83f6f358d8f66f3311c859754f.jpg': [4, 4, 4, 4, 5, 4, 1], 'D-868-7_140_2msX-10-_jpg.rf.84fcfc4d0db19bb958cd1e0868fd6487.jpg': [4, 4, 2, 2, 2, 3, 3, 3], 'cropped_B1330-5_56-0_2msX-48-_jpg.rf.e43b3bb921ba68a829d03cf753ff7ad9.jpg': [4, 3, 2, 2, 3, 2, 3], 'cropped_B1330-5_56-0_2msX-82-_jpg.rf.08d11dc2c7e889c8e8d440c0630e2d55.jpg': [4, 4, 1, 1, 4, 1, 1], 'cropped_C-868-_7140_2msX-102-_jpg.rf.a4969bbb3bea8d7d2e77c737cd63b502.jpg': [5, 4, 4, 4, 3, 3, 4, 5, 4], 'D-1700-4_600_5msX-11-_jpg.rf.493c7a098afe903c2e76abe7dbbca8b6.jpg': [3, 2, 4, 1, 1, 1, 1], 'cropped_C-1300-5_56-0_2X-114-_png.rf.30b891c495e072cb9eb3e66711bd1eb0.jpg': [2, 5, 2, 2], 'cropped_B-1052-6_40-0_2msX-39-_jpg.rf.ce3139129e56a83c90b7c6f43eb19e1a.jpg': [3, 1, 1, 2, 4, 2, 2], 'cropped_C-1300-5_56-0_2X-127-_png.rf.81c5863be04a86902d4f62e1da075491.jpg': [4, 4, 2, 1, 1], 'C-868-_7140_2msX-151-_jpg.rf.2e14b86adb8d8cc9963db8da7ab1664f.jpg': [4, 4, 4, 4], 'B1330-5_56-0_2msX-9-_jpg.rf.6e01cf75b14f9a274e879cb5ae5520fa.jpg': [4, 4, 2, 4, 2, 1, 1, 1, 1, 1], 'C-868-_7140_2msX-31-_jpg.rf.dcadf0f9108f5e8372496654d0805fb3.jpg': [3, 1], 'cropped_C-1700-4_600_2mmX-127-_jpg.rf.3ee23360f53ed6d0e6b473f41e745494.jpg': [3, 2, 2, 4, 4, 4, 5, 5, 4, 3, 1], '-0_2X-212-_jpg.rf.0bbcefb58dc9f9b5b384e5803edc742c.jpg': [4, 3, 2, 2, 3, 2, 3], 'cropped_B1330-5_56-0_2msX-82-_jpg.rf.a412d6a0e4983ca787826f7099df8451.jpg': [3, 1, 1, 1, 5, 2, 3, 5, 2, 2, 2, 2, 2, 4, 4, 4, 5, 1, 1, 1], '-0_2X-18-_jpg.rf.6227f152e1b6cdf390ac62dbba3bb466.jpg': [4, 4, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'C-868-_7140_2msX-107-_jpg.rf.b82a00dc25955809bf53675beb29b12e.jpg': [5, 4, 4, 4, 4, 2, 2, 4, 4, 3, 3, 3], 'B1330-5_56-0_2msX-89-_jpg.rf.c9b71d4e0be774c3cacd72f70dd2fcce.jpg': [1, 1, 4, 4], 'C-868-_7140_2msX-7-_jpg.rf.cd67748f26c87aaf35ac4459542a3af2.jpg': [5, 4, 4, 4, 2, 1, 5, 2], '868kg0_2msX-402-_jpg.rf.d2fc762614248d5d127d8acffd96077b.jpg': [2, 4, 2, 3, 3, 1], 'cropped_C-1300-5_56-0_2X-173-_png.rf.9044a61c5c987175be3393cf8bb23cd6.jpg': [4, 2, 2, 2, 2, 2, 2, 2, 5, 1], 'B-1052-6_40-0_2msX-31-_jpg.rf.16dcdaf1cd834b55ec82969bd65a1d4c.jpg': [5, 4, 5, 5, 1, 2, 2], '868kg0_2msX-128-_jpg.rf.07eff12391190d08350beeef0e26c4ec.jpg': [4, 2, 2, 1, 2], 'B-1052-6_40-0_2msX-11-_jpg.rf.53dd8d8cfc59c582b6140a464b28214b.jpg': [4, 4, 3, 2, 3, 5, 2, 2, 1, 1, 1, 2, 2, 2, 4, 2, 4, 4, 3, 3, 1, 1], 'C-1300-5_560_2msX-71-_jpg.rf.6a8b66e8fec7b566fb7648f7db68318b.jpg': [4, 4, 5, 5, 5, 1, 4, 3], 'cropped_B-1300-5_56-0_2mmX-157-_jpg.rf.f0287be83b0aaff4a317a49364976625.jpg': [2, 4, 2, 4, 4, 5, 5, 3], '-0_2X-16-_jpg.rf.1e2c432ea785a0075abc7b8597810d65.jpg': [5, 5, 4, 3], 'cropped_D-1700-4_600_5msX-37-_bmp.rf.42087b78550c1e792871f0f53f30d704.jpg': [4, 4, 5, 5, 1, 4, 3, 2, 2], 'cropped_B-1300-5_56-0_2mmX-147-_jpg.rf.42d91d6e298e805c6ff69527c724b62a.jpg': [4, 2, 2, 2, 1], 'cropped_B-1052-6_40-0_2msX-27-_jpg.rf.0a2e924e27e03f52be1c2024e87e5257.jpg': [4, 4, 2, 1, 1], 'C-868-_7140_2msX-151-_jpg.rf.299a905bb60265a4c2e176683b32ecbe.jpg': [5, 4, 4, 4, 2, 1, 5, 2], '868kg0_2msX-402-_jpg.rf.e12db7f63cca6dbcd8b67beb7088f9fc.jpg': [4, 4, 4, 4, 2, 2, 2, 3, 3, 3, 4, 5, 1, 2], 'B1330-5_56-0_2msX-91-_jpg.rf.eb8687e4ecfd58a12bfa8f1ed1ba981d.jpg': [3, 1, 1, 1, 2, 4], 'cropped_C-1300-5_56-0_2X-96-_png.rf.1dfa1378ca5b4e23e776feb85a5bc5e4.jpg': [5, 4, 4, 4, 2, 2, 2, 3, 3], 'B1330-5_56-0_2msX-92-_jpg.rf.ca4e7cdf8bf3bb1d30f2924cdebca8a2.jpg': [1, 4, 4, 4, 4], 'cropped_D-868-7_140_2msX-23-_jpg.rf.252ef329f87702e9f16d2547f732e2e9.jpg': [3, 2, 2, 5, 4, 3, 3, 3, 1, 1, 1, 1, 5, 5, 5, 5], 'cropped_B-1700-4_60-0_2X-315-_jpg.rf.887c0823f028593b1ae3b4dd7efa86a8.jpg': [2, 2, 2, 1, 1, 4, 1, 1, 1, 3, 3, 3, 3, 5, 2, 5], 'C-1300-5_560_2msX-73-_jpg.rf.d152597ab1d53517a0fbaed71f66403d.jpg': [1, 1, 1, 5, 4, 4, 5, 4, 5, 5, 1, 1, 1], 'D-868-7_140_2msX-69-_jpg.rf.d0224d607464059f96065c121219c103.jpg': [5, 4, 1, 3, 3, 1, 1, 3, 3, 2], 'C-1300-5_560_2msX-172-_jpg.rf.67ad622eee707c6d22e104b017f3d925.jpg': [1, 5, 4, 1, 3], 'C-1700-4_600_2mmX-97-_jpg.rf.78af6f132daa01cc31b2be1d58af1d4e.jpg': [2, 1, 4, 5, 5, 3, 4, 1, 3], 'C-1300-5_560_2msX-113-_jpg.rf.33d92cac3b97865dcea44ad03c71dea2.jpg': [4, 4, 4, 1, 5, 5, 4, 4], 'D-1700-4_600_5msX-43-_jpg.rf.6eb28c2810f2be3cfe5c929e287923fd.jpg': [2, 2, 2, 2, 2, 3, 4, 4, 5, 5, 2, 3, 1, 1, 3, 3, 4, 1, 1, 1, 1, 1], 'C-1300-5_560_2msX-66-_jpg.rf.ccecc00e4293fa662764079f29a44647.jpg': [1, 4, 5, 4, 5], '868kg0_2msX-11-_jpg.rf.d1b96b8bd3e84bc1e86026f35d1b67cb.jpg': [3, 1, 1, 2, 4, 1, 4], 'cropped_C-1300-5_56-0_2X-171-_png.rf.51f021c39237289d99cd8c2c9063ae48.jpg': [4, 5, 1], 'B-1052-6_40-0_2msX-8-_jpg.rf.6041163e93b6f57f038ac3c78256c0ff.jpg': [1, 4, 2, 4, 3, 3, 3, 5, 4, 1, 1, 1, 5, 5], 'C-1300-5_560_2msX-105-_jpg.rf.e0d6d8e08f95e2765afbf35a6f14e312.jpg': [2, 4, 4, 1, 1, 1, 1, 1, 1, 1, 2], 'C-868-_7140_2msX-104-_jpg.rf.bf8ea1c95cdf0dc3eb7cf89c9fdc13eb.jpg': [3, 2, 1, 1, 1, 4, 1, 3, 5, 5]}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "---\n",
        "\n",
        "# **Generate Colored Multi-Class Segmentation Mask Using COCO Annotations**\n",
        "\n",
        "\n",
        "---\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "K-g5_B9XJQz8"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install pycocotools\n",
        "\n",
        "import numpy as np\n",
        "import cv2\n",
        "from pycocotools import coco\n",
        "\n",
        "def create_colored_multi_class_mask(image_id, annotations, image_size=(256, 256)):\n",
        "    # Initialize a blank RGB mask (3 channels: R, G, B)\n",
        "    mask = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)\n",
        "\n",
        "    category_colors = {\n",
        "        1: (0, 255, 0),    # Compression Crack - Green\n",
        "        2: (255, 0, 255),  # Rebar Detachment - Purple\n",
        "        3: (255, 0, 0),    # Shear Type - 01 - Red\n",
        "        4: (0, 255, 255),  # Shear Type - 02 - Cyan\n",
        "        5: (255, 165, 0)   # Tension Crack - Orange\n",
        "    }\n",
        "\n",
        "       relevant_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]\n",
        "\n",
        "    # Create a COCO object\n",
        "    coco_instance = coco.COCO()\n",
        "    coco_instance.dataset = annotations\n",
        "    coco_instance.createIndex()\n",
        "\n",
        "    for ann in relevant_annotations:\n",
        "        # Each annotation has a category_id which corresponds to a crack type\n",
        "        category_id = ann['category_id']\n",
        "\n",
        "        # Get the segmentation mask for the annotation\n",
        "        ann_mask = coco_instance.annToMask(ann)  # Use the coco_instance\n",
        "\n",
        "        # Resize the mask to match the image size (256x256)\n",
        "        ann_mask_resized = cv2.resize(ann_mask, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)\n",
        "\n",
        "        # Apply the corresponding color to the mask\n",
        "        mask[ann_mask_resized == 1] = category_colors[category_id]\n",
        "\n",
        "    return mask"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mJURm3rF_uLf",
        "outputId": "33cbb8b7-dba5-4244-8fad-915dd081b1d5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: pycocotools in /usr/local/lib/python3.11/dist-packages (2.0.8)\n",
            "Requirement already satisfied: matplotlib>=2.1.0 in /usr/local/lib/python3.11/dist-packages (from pycocotools) (3.10.0)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from pycocotools) (2.0.2)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=2.1.0->pycocotools) (1.3.2)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=2.1.0->pycocotools) (0.12.1)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=2.1.0->pycocotools) (4.57.0)\n",
            "Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=2.1.0->pycocotools) (1.4.8)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=2.1.0->pycocotools) (24.2)\n",
            "Requirement already satisfied: pillow>=8 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=2.1.0->pycocotools) (11.1.0)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=2.1.0->pycocotools) (3.2.3)\n",
            "Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=2.1.0->pycocotools) (2.8.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.7->matplotlib>=2.1.0->pycocotools) (1.17.0)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "---\n",
        "\n",
        "# **Display Original Image with Colored Multi-Class Mask**\n",
        "\n",
        "---\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "wAN5zBBWJu_d"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Define the image directory path\n",
        "from tensorflow.keras.utils import load_img\n",
        "import matplotlib.pyplot as plt # Import the pyplot module\n",
        "image_dir = '/content/drive/My Drive/CrackDetection/train/images/'\n",
        "\n",
        "# Load annotations from the JSON file (assuming 'annotations' variable already holds this data)\n",
        "train_annotations = annotations  # Assign the annotations data to train_annotations\n",
        "\n",
        "# Extract image IDs from the training annotations\n",
        "image_ids = [img['id'] for img in train_annotations['images']]\n",
        "\n",
        "# Now you can access the first image ID\n",
        "image_id = image_ids[10]\n",
        "\n",
        "# Get the multi-class mask\n",
        "colored_multi_class_mask = create_colored_multi_class_mask(image_id, train_annotations, image_size=(256, 256))\n",
        "\n",
        "# Display the image and multi-class mask\n",
        "plt.figure(figsize=(10, 5))\n",
        "\n",
        "# Display original image\n",
        "plt.subplot(1, 2, 1)\n",
        "image_info = train_annotations['images'][image_id]  # Fetch image information using the image_id\n",
        "image_path = os.path.join(image_dir, image_info['file_name'])  # Create the full image path\n",
        "image = load_img(image_path, target_size=(256, 256))  # Load and resize the image\n",
        "plt.imshow(image)\n",
        "plt.title('Original Image')\n",
        "\n",
        "# Display multi-class mask with colors\n",
        "plt.subplot(1, 2, 2)\n",
        "plt.imshow(colored_multi_class_mask)  # Display the colored multi-class mask\n",
        "plt.title('Colored Multi-Class Mask')\n",
        "plt.colorbar()  # Optional: To visualize the class colors\n",
        "plt.show()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 471
        },
        "id": "zBhHRksc_xeA",
        "outputId": "2709243b-53cd-40d6-c842-fd2e11e2251e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "creating index...\n",
            "index created!\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x500 with 3 Axes>"
            ],
            "image/png": "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69bBarUO2nfSpElF97OlpWXQtsrKSvT29or3uVwOv/rVr3DPPfdgzZo1yGaz4rPq6uqizzWS/lRUVMDtdqOmpmbQ9u7ubt22Bx98EHfccQc++eQTpNNpsZ2Pz7p169DY2Aiv16s7Vh6zVatWQdM03HjjjbjxxhsN+9rZ2alEkEKhUCgUuxDBgLUkIqhc2XOvXLHLUFFRgcbGRrz33nsF93vvvfcwZswYBINB3XaPx7MjuycwqxinsTykH//4x7jmmmtw5JFH4qGHHsLf/vY3LF26FHvttRdyudwO708xfXzooYdwwQUXYOLEibj//vvxwgsvYOnSpZg7d+6I+kjHXHvttVi6dKnhazhiU6FQKBQKxY4nq+VK9iqWRYsW4cADD0QgEEBdXR3OOOMMrFixQrfP0UcfDYvFonvJUT/r16/HySefDK/Xi7q6OnznO99BJpMZ1vUrT5Bil+CUU07Bb3/7W/zzn/8UFd44//d//4e1a9fi61//+ojab21tRS6Xw5o1azB58mSxfdWqVSPusxFPPvkkjjnmGNx///267eFweJCHZrR48sknMWHCBDz11FO6CnoLFy7U7dfa2opXXnkFsVhM5w2Sx4xKljscDsybN28H9lyhUCgUCkWpyEFDDuYFpYbTTrG8+uqruOyyy3DggQcik8ng+uuvx/HHH4+PPvoIPp9P7HfxxRfj1ltvFe+5HZLNZnHyySejoaEBr7/+Otra2vCVr3wFDocDP/7xj4vui/IEKXYJvvOd78Dj8eDrX//6oNCtnp4efOMb34DX68V3vvOdEbU/f/58AMA999yj237XXXeNrMMm2Gy2QRXqnnjiCWzatKmk59keyFvE+/nmm2/ijTfe0O03f/58pNNp/Pa3vxXbcrkc7r77bt1+dXV1OProo3Hfffehra1t0Pm2bNlSyu4rFAqFQqEoU1544QVccMEF2GuvvTBr1iwsXrwY69evx/Lly3X7eb1eNDQ0iBePAvr73/+Ojz76CA899BD23XdfnHjiibjttttw9913D6vinfIEKXYJJk+ejAcffBBf+tKXMHPmTFx00UUYP3481q5di/vvvx9dXV344x//iIkTJ46o/dmzZ2PBggW488470d3dLUpkr1y5EgAGrSk0Uk455RTceuutuPDCC3HooYfi/fffx8MPP1xwgdedzSmnnIKnnnoKZ555Jk4+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Llxo2p+hOPPMMzUAWm9v75D7mrUbi8UG7Td//nxtwoQJ4v3TTz+tAdDefvtt07avvPJKLRgMaplMpqi+cGhce3p6NKfTqf3hD3/QNE3TlixZolksFm3t2rW6e1So74sWLdIsFou2bt06TdM0rbe3VwOg/exnPyvYh9bWVu3kk0/WNE3TfvWrX2kWi0W77bbbhn0t5Qb9TXj+vfHaP9ZM3O7X8++NL8v/CyquQaFQKBRDorxA5UskEgEABAKBovb/61//CgC45pprdNu//e1vA0DB3KG//vWvaGhowDnnnCO2ORwOfOtb38LAwABeffVV3f4LFiwQIWIA0NPTg5dffhlf+MIX0N/fj66uLnR1daG7uxvz58/Hp59+ik2bNolzHXLIITjooIPE8bW1tfjSl75U1HVyzj33XKxatQpvv/22+GkUCrcrMNz7aQT3lvX19aGrqwtHHXUUPvvsM/T19QEAQqEQAOAvf/kL0um0YTuhUAjRaBRLly4dcV8qKytxwgkniOIUjzzyCA499FC0trYO2fdoNIquri4ceuih0DQN//nPf8Q+TqcTy5YtQ29v75B9uP3223HllVfipz/9KW644YYRX0u5kYMFOVhL8CrP/w9KBCkUCoWiKDRVGKEsCQaDAID+/v6i9l+3bh2sVismTZqk297Q0IBQKIR169YVPHby5MmDcsemT58uPueMHz9e937VqlXQNA033ngjamtrda+FCxcCyIdp8XPJTJ06tajr5Oy3336YNm0aHnnkETz88MNoaGjA3Llzh91OKenp6UF7e7t4kTgZ7v004rXXXsO8efPg8/kQCoVQW1uL66+/HgDEeY466igsWLAAt9xyC2pqanD66afjgQce0OWFffOb38SUKVNw4oknYuzYsfjqV7+KF154Ydj9Offcc0Wo2jPPPFNQgK5fvx4XXHABqqqq4Pf7UVtbi6OOOkrXd5fLhZ/+9Kd4/vnnUV9fjyOPPBK333472tvbB7X36quv4nvf+x6+973v7RF5QJw9PSdIiSCFQqFQFIXyBpUnwWAQTU1N+OCDD4Z13M6433L+DhU1uPbaa7F06VLDlyzOSsW5556Lxx57DI888gi++MUvmhYBMRuXbDZb0v6cddZZaGxsFC/KsZk2bRoA4P333x9Ru6tXr8axxx6Lrq4u/OIXv8CSJUuwdOlSXH311QC23QOLxYInn3wSb7zxBi6//HJs2rQJX/3qVzF79mwMDAwAAOrq6vDuu+/i2WefxWmnnYZXXnkFJ554Is4///xh9em0006Dy+XC+eefj2QyiS984QuG+2WzWRx33HFYsmQJvve97+GZZ57B0qVLsXjxYl3fAeCqq67CypUrsWjRIrjdbtx4442YPn268BYRe+21F6ZOnYo//OEPWLNmzbD6rShvlAhSKBQKxZCQ4aeEUHlyyimnYPXq1XjjjTeG3Le1tRW5XA6ffvqpbntHRwfC4bBpmBId++mnnw6q0EaV1wodCwATJkwAkA+hmzdvnuGLwsDoXDIrVqwY8hqNOPfcc9HW1oaVK1cW9ERUVlYCAMLhsG57IQ8ZMZzvzx133KETf1TC+9RTTwUAPPTQQ0W3xXnuueeQTCbx7LPP4utf/zpOOukkzJs3z7SgxCGHHIIf/ehHeOedd/Dwww/jww8/xKOPPio+dzqdOPXUU3HPPfeIxXh///vfY9WqVUX3yePx4IwzzsCyZctw3HHHoaamxnC/999/HytXrsQdd9yB733vezj99NMxb948NDU1Ge4/ceJEfPvb38bf//53fPDBB0ilUrjjjjt0+9TU1ODFF1+Ew+HAsccei82bNxfd73KHCiOU4lWOlGevFQqFQrHToMpwSgCVL9/97nfh8/nwta99DR0dHYM+X716tShrfNJJJwEA7rzzTt0+v/jFLwCgYMW0k046Ce3t7XjsscfEtkwmg7vuugt+v1+ELZlRV1eHo48+Gvfddx/a2toGfb5lyxbduf71r3/hrbfe0n3+8MMPFzyHGRMnTsSdd96JRYsW6fKMjPYDgH/84x9iWzabxW9+85shz+H1egEMFlBGzJ49Wyf+ZsyYAQCYM2cOTjjhBPzud7/DM888M+i4VCqFa6+91rRdm80GQB/e2tfXhwceeEC3X29v76AQ2H333RcAREhcd3e37nOr1Sqq3RVTTp1z7bXXYuHChbjxxhuH1XdN0waV5I7FYoMqy02cOBGBQMCwX2PHjsWLL76IeDyO4447btB17a7kc4JK8ypHVIlshUKhUAyJEkDlzcSJE0WY1/Tp0/GVr3wFe++9N1KpFF5//XVRxhoAZs2ahfPPPx+/+c1vEA6HcdRRR+Gtt97Cgw8+iDPOOAPHHHOM6XkuueQS3HfffbjggguwfPlyjBs3Dk8++SRee+013HnnnUUl89999904/PDDMXPmTFx88cWYMGECOjo68MYbb2Djxo3473//CyAv7P7whz/ghBNOwJVXXilKZLe2tuK9994b0TgVU9Z5r732wiGHHILrrrsOPT09qKqqwqOPPopMJjPksR6PBzNmzMBjjz2GKVOmoKqqCnvvvbfhejiF+P3vf4/jjz8eZ511Fk499VQce+yx8Pl8+PTTT/Hoo4+ira3NdK2g448/Xnhvvv71r2NgYAC//e1vUVdXpxOeDz74IO655x6ceeaZmDhxIvr7+/Hb3/4WwWBQCOWvfe1r6Onpwdy5czF27FisW7cOd911F/bdd1+RB1Yss2bNwqxZswruM23aNEycOBHXXnstNm3ahGAwiD/96U+Dih+sXLkSxx57LL7whS9gxowZsNvtePrpp9HR0YGzzz7bsO1Jkybh73//O44++mjMnz8fL7/8ssi/UuyeKBGkUCgUiqLQNE0VRyhjTjvtNLz33nv42c9+hj//+c+499574XK5sM8+++COO+7AxRdfLPb93e9+hwkTJmDx4sV4+umn0dDQgOuuu04UJzDD4/Fg2bJl+P73v48HH3wQkUgEU6dOxQMPPCBE1lDMmDED77zzDm655RYsXrwY3d3dqKurw3777YebbrpJ7NfY2IhXXnkFV1xxBX7yk5+guroa3/jGN9DU1ISLLrpoRGNULA8//DC+/vWv4yc/+QlCoRAuuugiHHPMMTjuuOOGPPZ3v/sdrrjiClx99dVIpVJYuHDhsEVQbW0tXn/9ddxzzz147LHH8IMf/ACpVAqtra047bTTCoq5qVOn4sknn8QNN9yAa6+9Fg0NDbj00ktRW1urWwiVxO+jjz6Kjo4OVFRU4KCDDsLDDz8sClp8+ctfxm9+8xvcc889CIfDaGhowBe/+EXcfPPNO2RhZYfDgeeeew7f+ta3RK7PmWeeicsvv1wnoJqbm3HOOefgpZdewh/+8AfY7XZMmzYNjz/+uOECs8TMmTPx/PPPY968eTj11FPxwgsvDHvdqXIiByuyJQgKy6E8/y9YtDL8jxaJRFBRUYG+vj6l0hUKhWInoWkaIpEIQqGQ+vurUCgUZQrZ0Y++OwPegG2724v1Z3H2vh+V3f8FlROkUCgUCoVCoVAo9ihUOJxCoVAoCkJhcCovSKFQKHYfaLHT7W+n7ILKACgRpFAoFAqFQqFQ7HFkNQuy2vZPbpWijdFAhcMpFAqFYkgsFosqjKBQKBSK3QYlghQKhUJRNLu7CLr77rsxbtw4uN1uHHzwwbo1aBQKhWJ3Iru1OlwpXuVIefZaoVAoFDuNPWWh1MceewzXXHMNFi5ciH//+9+YNWsW5s+fj87OztHumkKhUJScnGYt2ascUTlBCoVCoVAA+MUvfoGLL74YF154IQDg17/+NZYsWYL//d//xfe///0hj8/lcti8eTMCgcAeIRoVCsXOR9M09Pf3o6mpaYesxbQnoUSQQqFQKIrCYrHstv90U6kUli9fjuuuu05ss1qtmDdvHt544w3DY5LJJJLJpHi/adMmzJgxY4f3VaFQKDZs2ICxY8duVxulCmXLlml1uN3zv5lCoVAoSs7uHBbX1dWFbDaL+vp63fb6+nq0t7cbHrNo0SJUVFSIlxJACoViZxEIBLa7jRy2VYjbnldu+y9nVFAiSKFQKBRFs7sXRhgO1113Hfr6+sRrw4YNo90lhUKxh7C7TkjtTFQ4nEKh2C3gxrn651Ba9gThU1NTA5vNho6ODt32jo4ONDQ0GB7jcrngcrl2RvcUCoWi5JRusdTy9KmUZ68VCoXCgEwms0cY7IrS43Q6MXv2bLz00ktiWy6Xw0svvYQ5c+aMYs8UCoVix5DVrCV7lSPKE6RQKMoeTdOQy+Vgs9mUF2gHszuP7zXXXIPzzz8fBxxwAA466CDceeediEajolqcQqFQKHYflAhSKBRlD3l/dufE/dGExlTTtN3a0/bFL34RW7ZswU033YT29nbsu+++eOGFFwYVS1AoFIrdgRwsyGH7/2eWoo3RQIkghUJR9uyuZZt3FXZn4SNz+eWX4/LLLx/tbigUCsUOp1ShbOUaDleevVYoFArFToXyrfYkQaRQKBSK3RflCVIoFAqFQqFQKPYwSrdYann6VJQIUigUCkXRKE+QQqFQ7B7kNAtyWglygkrQxmhQntJNoVAoFDsdqsKnUCgUCkW5ozxBCoVCoRgSm82GXC6HVCo12l1RKBQKRQnIlSgcrlwXS1UiSKFQKBRDQuXHlSdIoVAodg9ymhW5ElR2K0Ubo0F59lqhUCgUOxXKBVI5QQqFQqHYHVCeIIVCoVAUhVqIVqFQKHYfsrAgW4KFTkvRxmigPEGKHYamabj++uvx8MMPI5VKIZPJqFAahaJMoXC4bDY72l1RKBQKRQmgcLhSvMoR5QlS7FBWrVqFqqoqYThZreX5RVEo9mS4B0iFwykUCoVid0CJIMUOw2Kx4PHHH0c2m0U2m4XValXhNApFGcKFjxJBCoVCsXuQRWlC2co1PkCJIMUOx2q1Kg+QQlHmaJoGi8WiRJBCoVDsJqjqcArFDkLTNF0ekMoHUijKF1UdTqFQKBS7E8oTpNihUBic1WpFLpcTydUKhaK8UCJIoVAodi+ymhXZEnhxStHGaKBEkGKHYbFY4HQ6xXu7XT1uCkW5YrVaoWkaYrHYaHdFoVAoFCVAgwW5EuQEaWVaIltZpYodivL6KBQKhUKhUCh2NZQIUigUCsWQqMIICoVCsXuhwuEUCoVCoSgCJYIUCoVi9yGnWZDTtj9ipxRtjAblKd0UCoVCsVOh4ibJZHK0u6JQKBQKxXajPEEKhUKhKArlCVIoFIrdhyysyJbAH1KKNkaD8uy1QqFQKHYqKidIoVAodi8oHK4Ur2JZtGgRDjzwQAQCAdTV1eGMM87AihUrdPskEglcdtllqK6uht/vx4IFC9DR0aHbZ/369Tj55JPh9XpRV1eH73znO8hkMsO6/rIWQZqmqZf0KmaMzMbS7Dij93zbcO5RoWPN9inU/+H0Yah2C/VzuNc9VB+K+Wwk97vQccW2OxKG8xyYXfdQz9tQ5xrJdRQzTiPtj0KhUCgUCj2vvvoqLrvsMvzrX//C0qVLkU6ncfzxxyMajYp9rr76ajz33HN44okn8Oqrr2Lz5s0466yzxOfZbBYnn3wyUqkUXn/9dTz44INYvHgxbrrppmH1peThcDfffDNuueUW3bapU6fik08+AZBXd9/+9rfx6KOPIplMYv78+bjnnntQX18/7HP94Ac/gMPhQC6XQyaTQSaTgcViwZQpU+B0OpHJZNDf3y8W7NQ0Del0GlZrXvtpmgaHwwGv14tcLidugMPhgM1mE4rSbrfD6XSKNoBtpZ9TqRT6+/uxfPly2O12sRioz+dDVVUVAoEAPB4PqqqqkEwmEY/HAQAulwtOpxMulwsA4PF44HQ64XA44HK5YLfb4XA4xMtms8FqtcLn84lj7XY7PB4P7HY7bDabaIuXpc5kMuL6qX8ymqYhk8nAarXCZrMByD9gAMRY5XI5aJoGm82mMwhpTMzazeVyg/pEn6XTadF3I4yMS5qJpp+5XA65XE6MT6F2qA/0PpfLiWPkz/h7WuSVPze8H9uLfE7als1mddvonHRe+ozvQ/dJvgb+7NL1yPsMdT8L9d/oGuTP6byF7jf1gd7TvZXvj9xH2pc/M8WuSyU/z0bPKv8pn9eoP7sj9PylUqnR7opCoVAoSkAOVuRK4A8ZThsvvPCC7v3ixYtRV1eH5cuX48gjj0RfXx/uv/9+PPLII5g7dy4A4IEHHsD06dPxr3/9C4cccgj+/ve/46OPPsKLL76I+vp67Lvvvrjtttvwve99DzfffLNujcpC7JCcoL322gsvvvjitpMwY+Tqq6/GkiVL8MQTT6CiogKXX345zjrrLLz22mvDPs/mzZuFWMlkMuKfc0VFBdxuN9LpNCKRiDDwASCdTgtjicSK2+1GNptFX1+fWODTZrMhkUjAYrHA4XDA7XbDbrcLo57ai8fj6OnpwUcffSSMNYvFAo/Hg/r6elRUVMDn8yEYDKKvrw+9vb3IZrNC8JBYc7vdQtjYbDbxcjqdOoPX4/EIUWSz2eDz+cR5uQij6yMRRPeBjDxuBFssFqRSKZ2hz0UQJUSTkUiGsmyMAxDjQ1A7PIyGjrFarULQ0fVQ+9RXEmZ8HCwWC1wul074UBt0DB1n1iaNAzeajYzbQtuN3puJKRmz9o3a4sfIngkuGPhxRv00Ek2FrrPQtfD7L3tPCrVfKJyKjqNnqBjvktwfoz4Y9cPIqyOfn2+Tf9/TkCcFFAqFQlH+ZDULsiWo7EZtRCIR3XaXyyUm6M3o6+sDAFRVVQEAli9fjnQ6jXnz5ol9pk2bhpaWFrzxxhs45JBD8MYbb2DmzJk6B8r8+fNx6aWX4sMPP8R+++1XVL93iAiy2+1oaGgYtL0Ydbc9ZLPZQbPSNPPMZ/wTiQQACCPcZrOJWWr+z568RvQ5b5v2Jy9UOp1GOp0W50ilUvB6vbBarchkMkgkEmhra8PmzZuRTqdFv+x2O6LRKOx2uxBAmUxGGHL0PpvNIpvNwuFw6GaufT6f6Ad5fKh/JF64F4euhxvEVqsVyWTS0Hilz83e0zZ+Dzh8TLg3wOl0wufzwePxwO12iy8KjQOJPfKM+f1+Xb9J6NIxbrcbDodDjCE/1uVyCcFJHjPu7ePiqljxYzZWw2Eo74GRoS4b/rL3y+g6jAQQf4b49cvI4pVDzxg/3khomW2ThTjBPWBGbQ41zvIY8Gs2Euyyl6/QvS7EnuINUigUCoXCiObmZt37hQsX4uabbzbdP5fL4aqrrsJhhx2GvffeGwDQ3t4Op9OJUCik27e+vh7t7e1iHzmCjN7TPsWwQ0TQp59+iqamJrjdbsyZMweLFi1CS0tLUerOiGQyqSvLSkqTG4CpVArpdBqapiEWiwkBwIURGUdVVVXCKCajnAzmnp4e0Q6p11wuJ7xCZFxls1l4PB6dl4HDxQcP2+KGIwkSMvzpxWeq0+k0MpmM6ewriQxgsAFHIo6EgZHXgN7LhqhsfGazWd0MvZHRa2bs8usnA5fEYzgcFgLNYrEgnU4jlUrB4/EIgeJyucQx1Ab3mnGDHoAuMY76RcfK/eShYUBewGuaJoQn9SubzSKTyYgwRT4ePLyLP2vUV/L02Ww24YkcSnRRW/SM0DND95u8Y+l0WvST2ufiHciLCvKCAXnvpSxceKia3W4X1+dwOHQTCPTc2u12XWim0+lEIBAQn9HkgsfjEaKTi10SrSRYqQ0KESVPqcPhgMfjEf2idkn88rGTQ9mMvFLDEWfFQs/0SI4tR5QnSKFQKHYPSr1O0IYNGxAMBsX2obxAl112GT744AP885//3O4+jISSi6CDDz4YixcvxtSpU9HW1oZbbrkFRxxxBD744IOi1J0RixYtGpRnBOhzOsjII+M1mUyKnyRynE4ngsGguCnZbBapVEqEuSWTSeFN0DRNbCdjloxJOjaRSAiDm4sb3j/KaQDyoo36EYvFhIEfi8V0QkE2znjolpzzIedLyMKL/yRPCLBNKHDvDD8n/U4GMhnZRjPl/Di5L0bbedt0XSRa6TMubOTz8Xsi54FQf+XfaX+5z3LbFFJJ5+XH5HI5ITq42ClUjYTa52J2OB4DElHy+OVyOV1faR8+NrRN9grRfZG3A3lxRe3SPnJf6b7FYjFduKEcjsi9j/KLh47ySQGj37nYkdsx6he/Jvn7yI8zy1fj7RWD0fNH3036zlFeIc/x495J2pdENglCHhrrcrlE/0lU8+N4KC0PF3U4HIOen+FcH41VIpEQ4bsKhUKhKH80zYqctv05QdrWNoLBoE4EFeLyyy/HX/7yF/zjH//A2LFjxfaGhgakUimEw2GdXujo6BBRZg0NDXjrrbd07VH1OKNINDNKLoJOPPFE8fs+++yDgw8+GK2trXj88cfh8XhG1OZ1112Ha665RryPRCJobm4e5OHhxhcPASPjw+12w+v1wuFwCKOUGwU8AZ6ML248k1FBx5Awkg1xgs+s03u73Y5AIIBMJgOPx4OGhgYRDwnoPQl0jGzY8fblUB66Di6GyJNBM9bcEJRj/M0MdG44cgN6qFAr2i57m8w8SEYz6iSSzASV0bjz3+V+mB1nNL5GYzCc8Dd6Bui4kXoL5PEmgcaFpPzs8P3pd1mYGo0FF/RGn/Pn3+w+yeeSj5fFqlH7BO0rC2h5n2LGlYfvFbq24bQpH0/Xx8NtyYtFkyxc7JBwoc9oOxcy3Asmt8tFEBdJ9LvD4Rg0BnJf6b0cIsifq3Q6DYfDgffff39YY6JQKBQKBaFpGq644go8/fTTWLZsGcaPH6/7fPbs2XA4HHjppZewYMECAMCKFSuwfv16zJkzBwAwZ84c/OhHP0JnZyfq6uoAAEuXLkUwGMSMGTOK7ssOXyw1FAphypQpWLVqFY477rgh1Z0RZolVXLSQAcZnPcmgqKqq0s2SklFFXiIyVB0Oh/AMcfFCYUGUnE+5NyQs+MwoNxpcLhe8Xi/cbrcuD6mmpga5XA719fU44IADsHbtWt1sP7UzkploHr7FP6cQPwrpIiOJjEt5dleeoZfPI7+XDUtuQBYSHUYYXbcspIyEiJFRJ/dvJP3ZleDCwkj0AIUrtMnilLcle2aGwkjQcuTKe/w8ZscN57mXr2+obSNpd7jIY0rfe35+6pc8PqlUStdnLkL4/rwNI9FLOYcklOVxcLvdg8JTSQBRuCYJKT55EgwGRdixQqFQKMqfLCzIogSFEYbRxmWXXYZHHnkEf/7znxEIBEQkWEVFBTweDyoqKnDRRRfhmmuuQVVVFYLBIK644grMmTNHpM0cf/zxmDFjBs477zzcfvvtaG9vxw033IDLLrtsyBA8zg5fJ2hgYACrV69GY2OjTt0RsrobDnLlMqoQx/N8/H4/fD6fEDCJRAL9/f2IxWK6Km18ppbeEySSeO4HtU9hK7KRqWn58DMKWyHjkkJhLJZ88YTu7m7dLC4wPCPdyLDis8Pc0CQjJx6PC2OH523I+3OMDEojAbK9AkM2ImVvX6E+mW0jZFFXjshermLh3tFStsuPMfPqme1nJLZKJVR2BaFLzy6vKMmvn38uh+qRF5v/bZD3lwUO//7xcFEZmvCwWq0iL5L2o79ZNCFEfyPsdjuSyaSYWFIoFApF+ZPTSrVgavHnvPfee9HX14ejjz4ajY2N4vXYY4+JfX75y1/ilFNOwYIFC3DkkUeioaEBTz31lPjcZrPhL3/5C2w2G+bMmYMvf/nL+MpXvoJbb711WNdfck/Qtddei1NPPRWtra3YvHkzFi5cCJvNhnPOOacodTccyGBIJBK60s7JZFIIDgrBI6OLSl/TTKccvkJtaJo2qBIbX7OE8oaMjEfySvESzdQ+GR48vEwufGAkBHguCf+c50XJY0NJ7bygAb3IwOGCifqXyWTEddJsMr8++XplChmgZiLLKDyJG4rytZmdT/b+yB4j7lEqhmKM6WLaMmpnpAa//AzwELmR9o+3W+gY+XmX9y8UTmjWHvdqmB0ne0WKRR6XYo83EnAjhX9XC/WB9uMTO/JnZh5QjuxhMpo8MAt7o22y55X/LVCFERQKhUIxUor5P+x2u3H33Xfj7rvvNt2ntbUVf/3rX7erLyUXQRs3bsQ555yD7u5u1NbW4vDDD8e//vUv1NbWAsirO6vVigULFugWSx0J9M+bqonRP2oqWU2iRS65S0a+w+HQhcQRPJ+I3ywKi+OGg5EIovZ4+B1PEOcFCuSqcfx4gosXowpYcpgYF1xcSHADhudD8fWDqA/yzPNQnoJivDRm+w0lroyMPNmw3RmhTtsD99Rtj5eCXyf3GpCnbzhGvtn7QgLA7Hkvpl2jtsyETSGjv5i2zdqRw8h2lCfKqK3hCGoSQlzwy6LeSDwaeToLeUz5OMjiRvZcAdvyIBUKhUJR/uRKVBihFG2MBiUXQY8++mjBz4tRd8XCvRxkMFBCscPhQC6Xw8DAgIhtJ69LIpEYNHvODQESOpFIRJTOpjh62heATnAYGRG0L4WQkHFCniAev8+rvPFzcGOICx0+BoVyOGSjmfeTfo/FYiKsj66dvFNyQjq/7pHMsMuikhtsPCzHqN/yjDQfox2F0biORMTsjD4aeWNKjZHg3F4RysV7qQWt0XeyHERzKZC9hQqFQqFQcHKwIFeCnKBStDEa7PDCCDsSCmvjpazJc0EiIxaLiXVdMpmMWHMolUqJRGReLMFiyRcViMfjWLt2LVwuF4LBIKqqquB2u4WhlkwmEY/HkUqlEIvFRB+AbTlEPT09SCaTcDqdSCaTiEajyGQy6O/vF6WWOzo6BnlyAH18P3mNZGOXVwfj2zmymCBvEs3oUtnvZDIpiiaQaOPXJK+zI89Cm51/OHDvFM+/4kLQKDdCrhxXSiPXKCypWEFjdl+G47EZCj4OdC7ZU8gX0d2e81D7sheB2F4hVGrMQiPN9uH7yhMk5Qj9HZKLpAzn+HK9doVCoVAohqKsRRAPMSJPTzabRX9/vy6vhdYMymQyyGQyQryQF0aOoafiAfF4XFc1zePx6MpiJxIJpNNpIXS4gUELTtJaQvF4XBij0WhU9DUcDhuGrlF/qFgB9Y2H0fEFZIuBzwrbbDZdGW2+9g31n85rt9tFdTk63syoHEl/qE0uYOUwII6RUDTbt9QYiYyR7F+qfvJQSbM2+T6lOJ98Pdt7LTvSW1Gov2Z9L6WwH21G6g0qlbhVKBQKxa5LVrMgW4LFUkvRxmhQ1iKIh0iROEmn0+jq6tLludC6QOTJIFFEhQHIGCKBQQKJwtbS6TT6+voQj8eRSCSQzWbFDCutN+T1enWGJlWBI8M+lUqJ0Lp4PC5yl/r6+gZVhuOGMnm6yFNFxRwo78koZ4djNtPNvSo8zp/ey+uXyN4Lo3yEoYwsbnTyEDf5uo36yz+X8zlkYVGsQCsmb2l7Z8ONjjfqXyFxVOha+HdgOH3dHvE4HCFndJ7hip3tEUdmfS0k5EfiTdxeMW7Wj1IwlGgv9rtbyj4pFAqFYvRROUFlDIkALoSSyaRuO4V0cTFCIoBECjeeuQeCREImk9HlFFmtVni9Xlgs+ZLbVI2OPDP8OIvFIgRVRUUFgsEgnE4n1q5di3Q6Laq4Uf/kHCVgW8gTXR+V79a0/Dok1G/KO5JD4DiyyAK2eZdI1GlavtgEX32eQgHJo0V94hiJEu7hoVA8M0OKF2Wgn3TdvEiFy+XShc7J5x1O9Soe4lcMtC/3pNG5jYTN9nhKChnpvM/FhLoV47Eqdn2g4TDaxnOhsSnkCRrtcLhCkwPDEfvyNZjdY/l7wL2yBE2OqOpwCoVCodgdKGsRRN4WOZSLwuMA/eKB5BWSjXEjDwcwOP+Bh8bR73Qe8jYZeSmonXg8DgDwer1ChFFOjtGihoRR7o+8cCLP++CV3mQPCu8TvbihRIY9iT4KnePrD1E+UzEhVvL5+T0y8rgYGZ5U5Y9D+5I3T26/WIzGQf7c6P1Iw4yGYqjqcWb94UVCijlGfs75dRXTRikw84YVYjj9MNrXyCNktE+x4lV+5na0YDLzfsr7GP0dKyT2h+o3TSi5XK5B30WFQqFQlCc55Nf5KUU75UhZiyDK+ZHFgezNIU8C9/4QshFTaDaftyMXQeDnkr1LRCaTEYu08uINdJwsCuR+kBDhIXzcwOFhcWbGdCHPgpyXRC/KpyJBRGPIX0O1bTSuvI9mosJM2PB2t8fbwtsbTYyel+GIq2KN22KEwUgppr9mz4os0Hc0Zn018wwV09bOEkDFIo+1mWjnEwFGExPysaUU/QqFQqEYPbQSVYfTlAgaHchIJzFCXgEeGiavweN0OpHNZkV1OBIuskCi/XnoF3mfAIjzaVq+Ch0ZDdQfvviqy+USYWbhcFiU8ab1jXi4GMHXOaLzeb1eEZJCIVlcEFAInrxQq3xNNHb8erkIovOTAOJj7PF4RB5ToZXp5XCtQqLFrDBEIeOej/dwQ7mMziF7/vg1yMcV8twNdU4zZO9kMeFORn2UP+Mz90OJypEylPeK3/NihNBQ7e8owSE/m0MZ/YU8SYUYaf/NvKWF+mfm9RuqT7IASqfT4qVQKBQKRblT1iLI6XQC2JYbRGKFxAYZNJTkb7FYkEqlkEgk4HQ64Xa7RU4NAJ0nBRhslGqapss54qFiFHJHeUNcUFGuDUHihfKLqGgDIRvlJOyAfIU6OoY8W7QvnT+Xy+kKQVB7RmsRyYYpHcP3pdA/at9utw/ywnHMDEOjHBqjGWgSV7Q9k8nA5XINMgDNvGCFKCQCyDM1VAlrI2/NUKF03HM4HM8CjUUxYqhQSBzvk+ztK3T+Yig0BtSOUeGPQseMFkb3drjieqhnZ7gCaHu8d/LECj9WFkf8bxZ9D+jvi6ZpcLlcAKDC4RQKhWI3IaeVKBxOVYfb+fAQLjIUzQxqqrBG/9xpLRwK8zIyXOTiCSRIuNeFe4moTyRaZDEje0Z4G0bhUATvW1tbm/AexWIxIUbkamtGooqPDzdkZEHBtxsVAEilUuJ85HGiceJij/fBqG35dz4uRgJNHlN+Hr7d6DijazWjmNnyobwv/HfZsBzKuC7WA8Qxa3O4RvdwzmvmyTGC7tHOCqca6poLfT7S/hXj/TEL9xvOPTISamb7FCoMYeR15M+evP4W/dxRXjiFQqFQ7FxUdbgyRzbUeegV5QeR0c5zWXiOEF8IVG6bwt2ofQqzM5qdl0UIN+qN9iERVMhQkRfB3LJlC0KhEFwuFxKJBNxut2EYHz+WkL0oZoaNfE55TSESOuRlo1LiAHTFI7hRJQsbeayNwqX4Z7zvQxnfZkaa7Hkyu+fA0IUGijVoZa9LIaN1qHaLwex6zPYbieACRiaWzI7ZlbxBRs9SKfq3s4VDMSK9mGvdXsGmUCgUCsWuym4hgszWsSEhQkKD//OmhUxpH24YFDIEuNeDDH46hq/MbpajIveRhJvRdjLCySi2Wq1obm5GIBCA1WoVaxbJfeTvebvcqM9ms3A4HEN6Z0gw0jXLoX3UB/Ksud1uISr5+Mh9M/PkyN4xum6jtug8crt88VkjipnNLuQxKjTWZm1z4VXonPRzewxvs3tazP6lptwN5u0RpDuineFQjICRJx/kZ9AsjFGhUCgU5c+eHg5Xnv6rrVBSPrAtUV0uE02x7G63WxREIOPZLI+EG7K0kCgZAbRIKeUAyf3hhgQ30knU0DnJUOcCiEQGvZcLG9D5k8mkWPDV6Pwc8nZx48aoCAFB76mvPDSGFk51Op26PKVMJoNUKoVYLCYEDFWy49cpC00+JnybxWLRCSEKwaOFaVOpFFKpFDKZjK4wAxVxoP3482F0Pvle8/s/FMWII/k+yPdC9nDJbckhjmbnlY+TPX3F7FsqhnP+3YGhPHzyvvyYoRhqLMlLa/Tiz458LH8W+XeUh9bKVTbpXDtiLamdxc033zxoTKdNmyY+TyQSuOyyy1BdXQ2/348FCxago6NjFHusUCgUO5bc1upwpXiVI2XtCZJzQQDj0DPZyyK3IYsHIyOVflL1NTnEqVBpbH68HNYlz8Ryrw+wrdS1HC5lVJ2M513IY8C3j9RApeP4ukb8+km8kLHEDTG5v0YhblwMyeLFKFyOX5f8uVlOlHw9Rr8XCr0r5vhC+8nnMNpve703I723O3L/ndXW9jLSvhT6OyRvL/YchZ6b4QjNoTyx/G9ToX7uSvdppOy111548cUXxXs+mXP11VdjyZIleOKJJ1BRUYHLL78cZ511Fl577bXR6KpCoVAodjBlLYK4gczLUVM+B81uxmIx4ZmQ/9FTFTl6mc3UAxBGPv3ODXcelkazpvw4eRaYt81DTqgiE10TLe5K7dA/bVmEyOcyM965YOBiS96nEPIq8twjRVXpaGFFALriE2YV4uTPZC+XjNx/I0HIBSU3/Pj1U1vFXvtIkUVwMfub3V8jdgcDtRwxE8Kluh9DtVOMqJL7VSgH0ewY+jkcEbcrYrfb0dDQMGh7X18f7r//fjzyyCOYO3cuAOCBBx7A9OnT8a9//QuHHHLIzu6qQqFQ7HD29HC4shZBVKo5mUzqvAh9fX3in73dbhfChYokUAgIoBcTVFrabDaUe2CMwphkr4UsqChUhTwjcrgZkDfeU6nUIKMdgK5AAZ2be1iMPCM0Dvw4HpYnhwSaGet8DHkb8nvywFAoGhVN4IvWGoXAyaKQn99o4VcewkOhb4WEHx9Ped0c/rtc0a5U8Osbqoy13J9i+1HOxmm5syPG3iiMrdAEh9nx8t+xYsvJ0zn535ViRdSuyqeffoqmpia43W7MmTMHixYtQktLC5YvX450Oo158+aJfadNm4aWlha88cYbpiKIQpOJSCSyw69BoVAoSoUSQWUMrwQnexjonzaFr5HHgueJyPHtcqgZtUNGOxnwhFHYFDdQ6Dy5XA4NDQ0IhULw+/1CqEWjUfT39+uKM2SzWcTjcUORZRS2ZxZiJiMLJGpPDpeTw9K4ASXvIy9CK4s2Gn8SK7zcOJ2fiyij4zlGoUC8kp187408TkbjwtuWPUuyUC3kiTHqN99Ox/LKc8WENMkz8GZhc2bjpsTRNuS/EcT2jJH8/ZC3F+PJM+qjvAgwzy80usckUIye9aGEk5HXkX+3i+37rszBBx+MxYsXY+rUqWhra8Mtt9yCI444Ah988AHa29vhdDoRCoV0x9TX16O9vd20zUWLFuGWW27ZwT1XKBQKxY6grEUQN97lbfwzWtuH4IYF9zKQcSoXQ5DPKRubQ4WeaZqGxsZG1NXVIRgMAgB6enrQ1dWFVCqlEwA2mw3xeFxX8tvImCp0Xuqn3Af5WDMvFn8ZrcEje5X4uMpeDp6nxBdelL0zsuAc6jrlpG2z+2Rm9Bo9M2YUazAb9cPsGNlYNRNoZmFJZs9+Kb1XOwOj/u5oY3tHjNFQ91F+X6wYMptoGeo7YdSH4Vx3sX0sJ0488UTx+z777IODDz4Yra2tePzxx+HxeEbU5nXXXYdrrrlGvI9EImhubt7uvioUCsXOQHmCyhjyzmQyGTgcDgAQv/MwDpfLBZfLJcpKU8llCo0D9NW4qLxzIpFAJpMR5Z+TyaQwJlKplOgDHS8LLRJSTqcT++yzD6qrq+FyuWC325FMJrF27Vr09vbqQuOsVisGBgaGFQZVCLm0tJlxYySWZEHE25GPNTICubHOF1eln8C28aP36XTatOy3mdAxEgQyRgLIzDvEkQVzof3Nwtxk43Q43ju+BtRQ97AYih2jnYmRkNtRfSk0dkbeV/n9jhKZZt4i3l8jr6LZMfIkhdlxwGDvZ7kJ6ZESCoUwZcoUrFq1CscddxxSqRTC4bDOG9TR0WGYQ0TQ/xaFQqEoR5QIKmMonMpqtcLj8YjZdV7+1eFwIBAIwOfzwWKxoLu7Gy6XC8lkctA/fcohojAUOofdbofT6URlZSWSySTi8Tji8bgQUFQRjUOGHJWV7u3thdVqhcvlQkdHh2iLCh9wQ5lfQ7H5I4UoJhTG6Hczj4PRsbIQGsp7xQUReZxoezH9pjBHvu9QRpxZyFqxFDpGztWin3Lo20jOLY+/kYdud2FHCqChzlsMO9PbZhSuSxiFWMqfFTuRIof48r+HuzMDAwNYvXo1zjvvPMyePRsOhwMvvfQSFixYAABYsWIF1q9fjzlz5oxyTxUKhUKxIyhrEURlmUloUP4Jz0GhdW3IoPB4PLpjqB0AwkNEifYkcvjaG7RYKDeGjDwEcr4NeYXsdjtqamrgdrsxMDAwKMSskIAgZIN4qFCuQu8LfSav72MUqiYjXzONMV0fL58tL8JKhROcTqfYxg1+3j8+voXg1zBUSKB8DWbixcgjJR9rdg4u+GSGY1wbCS4613Du+VD7mRnexYb9FfO8yWFfxQrhofYr5P0wusdGkwH0nNLfGkBfWISHeBbqH+3DixsUenaN/qbI48Svw+gZNdrf6Frl9s3+DhmtW1VOXHvttTj11FPR2tqKzZs3Y+HChbDZbDjnnHNQUVGBiy66CNdccw2qqqoQDAZxxRVXYM6cOaoynEKh2G3RgJKs8VOu8QO7lQiiqmrANk+D3W4XpbE1TYPb7Rb78eptwLbqcSSCeGUzmtFPJpPieDIWjGZMKVSPoNLRVqsVVVVVSKfTOoOYcpHkinKFDA4urgqFysjIho4sKMhY42FstB+vhCT3hY8jvz/0O40pbeNrdPBQRPKA0fXxsTAK2zPyiPB7w716PAeMF2YwaksOReLXQ+fgPwt5zvjYyuLKTPQaiSvZ8yb30eiZKdQXftxwnp+RIguNYjxzZgKj0Hb+jJtdv/y88zHlx9DkCN+ff1eNrsHsXvAJDx7qONT310ygDSWAeB+Mvj9Gfabf+f7y34JyZOPGjTjnnHPQ3d2N2tpaHH744fjXv/6F2tpaAMAvf/lLWK1WLFiwAMlkEvPnz8c999wzyr1WKBSKHYcKhytz5NAgmqWlnB2n0wlg21o1fCaWcn/I8HY6nXC5XMITwQ0F2oeMeaDwjDiJJnqtX78efX19cDqd6OnpQTgcRjKZFKF39LvNZhMCiV+TzWaDy+XSrfKeSqUQi8VMc2gKbZNf1Gc+rkMZPIU8UmQ0pVIpMZtO40higkqc0+9utxuZTAbJZBLpdFr0yWazifWcCLnIgnyt1Dbta+YdMRJOdF+554uLp2KMQCPjlzAq+V0sZus6Dcc7Q8cY/W4m4uiz4YilofpitKAtkJ8woOdiOEa3LG6GevblXBj5+oC8eKfQWco15OtemY25mQfGTLjS3zCaGKC2uYin7XL1Rv6ZkfDhP3O5nMiLpO+H2+0W18rHndrh3lvqUzny6KOPFvzc7Xbj7rvvxt13372TeqRQKBSK0aTsRRD9c85kMiJULZVKwev1CuNX0zQhHMjLwI3qyspKuN1uOJ1O9PX1Ca9NOp2G1WoV4sjlciGXy2FgYADxeFwILWDocJiNGzeio6NDiJxEIgGLxQKHwyHOB2wzRAgKy3O73aioqNAZjk6nE5lMRrdAqVEfjIxBPqvLS1ebebXk9oy8IDJ0HZRrVVdXh2g0ilQqJcQkN7BIONK95AYdv5e8v7JBSSGL9Du1RWPL+yzvKxup3JtXSGSYYWSMk6HLPUJmFQD5Nm4Uc4OWe+zkJHi5Hf5erupH94GLcD5GvLT8cIWJfD5+XtkraXTtRucr5l6YiQH5d35PKCyT+kbPocvlEu9p/0KeGHkM+DXzZ0le+0cWZLyv/Hkxy/ehZ4FP9HAvl91uh9/vH7ROGTB4skAWvEZhqQqFQqEoX5QnqMyRDWR59p7P+vLFSmmby+VCMBiE2+2GxWJBOp0WhjOVr+ZGOuUYccOwUJ4MGTEURmf0mTyTb7QoKl0n34+2UQW44Yggo8+MDEXqn9yXQtdqhN1ux/jx44VHjDxfvH0aT9nwlT143FjjRiU3ALk44DP+Rp6MQh4D/l72dhXCyPiWPU382uheyoapkUEt94tf61BinB/D++N0OkXoKLAt58rlcomKfeQ15SLMrJ9G42Ak8Gg73SP+3SxEoWeQjy1vTxYVXFBSgRJ6TwVS6DrtdrsuD4jK7suiQx4L+X7ReQC9AJLHgNrhfeQilB9nNiHBRTL1kyZU+N9KPmbcO0vbHA6HuN5cLoeenp6C90ahUCgU5YESQWUMN3L5P3XZsJBzf8jYcTgcqK6uRigUgt1uRyqVQiAQEPvE43HEYjFRLpvPrJJXSF7nh/oln48XCMhms/B6vWJGlxtGwLb8Ge6JoHPJidh8NlsOnaHPZcxm3+XjCs36GgkIo/3pvdPpxKxZsxCPx4UASiQSYh+6bzSWPI+Hi0QSQWZQf8jgo9lvOg/3nGmaJsIlgW3lxKkdOQRONl7NrttIdMh949dEfaTnQH6GubDn95ueb248c6OYMAuh4/1zuVxwu93C2Kf7UFFRIYqFpNNpMXYUJsb7ZCbAZKMdgO4eyvcom83qRAa/L3K7RvBxI9HC88xkoUXeXjoXVXukyZBMJqPLZQOg+z7LZcyNRDYXIXROymPkzyKNC00GUJvy5A4PtyVxxM/PhT9/7mipAIfDIcJ7+d8YqqRJoX8Uhkv5kYlEAqtWrcI777xjOPYKhUKhUJQLZS2CKFSHYvSJdDqNcDgsjKuOjg7xz93r9WJgYECEtQSDQRE2JRvBfr8f8XhchMjU1tZi4sSJSKVS+Oijj9Db2yuMGO6F0bR8+J3P54PNZkM0GhWL8eVyOfT19SEWiwmvktPp1IkAaoP6QbO5yWRSNzNMBh4JMqqQV2hWWTbQeDvcsCSDnIwxGi+HwyFmhEnc8dA1HuJmseQ9aw0NDZg4cSK2bNkCi8UCt9uNRCKhM7gBiOOAvGiKx+NiLLkYIgOZ5zWQYeh0OkXOg9VqRTweR0VFBTRNQzQaFdfM13Ci0CbZM0U5SnTt9OIhcvRTFif8WaDwKh6OyceKjuEGNV03JeRzTyQdQ6GU5L2UPYR8jPj6Sz6fT4wBFaugZzWVSgnvD4Um0vNA942uO5VKwe12i++V1WoVlRPpOqkfyWQSlZWV8Hg8IrSUG/o8TLW3txderxdOp1M8b3R9drtd5AzxsXM6nfB4PGhvbxfPZWVlJaxWK9xut8h7CQaDOrHBJ0hIJNB9p/uXy+VQV1cn7in1ncaaRJPD4YDf79cJD7r3FFLLBSn/HvFJCfJC03imUilEIhFx3U6nE62trTpvKv2Nou+Tw+HQ5TwODAyIEGA+rul0Wnh/KCSQCzZ5EiYWi+G5557DM888A4VCoVCUN8oTVObQ7D4l0/Nqa9x4IuOChx7RNofDIYxLMryBbd4EKpQQj8fR1dUlDEczzwAAYVgD0IkCedabDEXu4eHeEB7qRJ9zI4v3gc4ley/4MdybwkWP7FniximVHge2FZgwelE/eH6Jw+EQhm9/fz+am5sxZswYRCIRbNy4EZFIRIw590DwtZq4B04WPbKxT8YcXYvf74fH40Emk0E8HteNFYcLFHpPQkO+x1arFYFAQCdIeT+56CFD1+VyiWeMG7xc5NjtdrhcLmGser1eMSNPL7fbrRuDVCqFVCqFaDQqiknQZzSjb7fbEY1GhWEfDAZ1XjG/34+KigrhmSAxlMlkxPpaJIJ42fJ0Oi1y6chbwHPt6DnLZDLYvHkzWlpa4Ha7hfjnzw6FmMbjcaxZswZVVVUIBAJCVPCx53lc9EzS2P3nP/9BJpOBx+PB+PHjAUD0LZvNwuPxiPvMQ11pnOie8TBXm80m+kHfbVqUmTxE9PzwUFn+HFH/6Lnk3znu/aP3tO5ZLBYTEwd0j5xOpxh3OoaH5tKzRfeIvNgkmulY+p74fD7dd42eE/L+0PZkMinC6RQKhUJR/miaBVoJBEwp2hgNyl4EETQjSrOxwDYRww1yboABEB4gngvBw4xcLpf4LJ1OIxKJ6HIhyAiXiweQwUuzwFx0yMYF957IXhzZ82OE7JGgdumnWYgU35+Xf+bhO9zDwceHG+7cCCRjkn7XNA1+v18Y1vX19fB4PKitrYXX60U4HEZ/fz/i8Tii0agwSGkGHIAIVaSx8Xg8utlr3l9+PywWC7xeLzweD5LJJLq6uhCLxXTjRu3YbDZ4PB6dAJVzKqjdXC6H2tpandDhAog8AzxUk4x/mm3nnjISC+QhITFDHhbK1eHhTAR5gvhMP93fQCAgKgpykRQIBKBpGlKpFJLJJEKhkPCQ8AqFJBromfV4PDqDncLE6Fq5B4mLgFQqhWAwiNbWVtF/CkOk/pI3NB6Pw2azoaamBsFgUAg2fs+oj3SPyENqs9nQ3t6OVCoFn8+H1tZWneeDvufUd15inzxZXIyTQJWFNQlZu90uRJXsCeSCiXsP5ckGfl2y2OPffavVKrxj5Bnl32P+rPHnn3tP5RBBaouELi8KAUAIcGqPJoyMcg8VCoVCoSg3yvq/GRk+FCJDhqbf7xfGr9frhcvlEmKHPALceBwYGBBGERmNDocDXq8X1dXVurCw3t5eDAwMIBaL6QSWXPSAZm6dTieSyaTIf7HZbAgGgwiFQkilUtiyZQsSiQTcbrfOoOGGEyEbg2Sc0mfy5zLcEDPyLHGPAxcalZWVYiY7FAoJw5JmlKurqxEIBITRSsa82+1Gb28vPvroI7z77rsiNNHr9aKurg6nnHKK8NJ88skn2LBhAyorKzF+/Hhs2bIFmzdvhsViweTJkxGJRNDZ2Yl4PI7Zs2eLsSeDjGa73W43GhsbhQHocrkQCAQwMDCADRs2oLu7W1yj3W4XRramaWhoaBDt0bNCn9EseywWw8DAACoqKnQz/Dy0S75XAHRGKxmjdE9IvHDvG5APN/P7/boQRZqll71Z/H5SG3KRBTontROPxzEwMID6+nqdiOTij5erpnGRjWOCh2ty0Wyz2dDa2joor4b6zcMLXS4XJk2aJMQV91zS+en5JXFMfU4mk2hsbBTfY5/PN6iKGxes9NzbbDYRAsc9ilRtkq6LT6JEIhHhAePPB8HFORUiIIFIEyQE90zz7yQXWgBE6C5dP5+w4e3J66dRkQfyGFosFvh8PvEd52MRi8WEB5HCfGm8KLSO5yApFAqFonzJwVKSxVJL0cZoUNYiiAy3dDotYtzJcEqn0wgGg2hubhY5E2TQ8JAxp9OJcDgsRAqtCULtkkFks9mQTCaFERgKheDz+YSBQEUUyAtEBjjN5MfjcWHA5XI59Pb2wuFwoL6+XhhSAHSLu2azWfT39+s8HZSDxA0XYFvuCRlAbrcbuVwOPp9P51EgI6axsVEkwtvtdnR1dQnDsbq6GnV1deJz8hzQtXGPCI05D4Hjs/RUfa+yslJUhQuHw1i5ciU+/fRTjB07Fg0NDaitrUVLS4sQrs3NzZg+fboIDaIxpr7z6nz8JefZWK1W9PT0IBaLQdM01NTUiPAoEqgAdDkR9J57NCj3iWb+KbSJe+3oPvDnk7ZxQ5znfZFXgxu9fFaeF8fgOWskVAj5vEYeBzKguZDiHhK5XS4++PeNvF38PBSqRzlBFEZI7cZiMTFmNJ682AOdO5lMor29HZWVleLZpTZpX7mMtcvlEqGF/Jknjy8P0eQeF+4poXsuf58ojIzGhM5NkyU0DrLHlKC/I9yjZBSOyceBnhHqJ78nZtfAr4N7kKj/tB8JHB7+R38XuMDk+Xl0j8lbpsLhFAqFYvdA5QSVOXL1MF7NibxCfAYzkUgIA4z2p7VryMijf/rkLeIiiM8e0/lpxpQqywHbZmO5Uc1DuCi23ufzoba2Vmdw8VC2rq4uMVNdUVEBwHwBRhJ/lA9C6yV5PB6dIaRpGgKBgBA5ANDY2Ci8KRMnTkRlZaUI26OZbG7M8/V3jEJ7KJG6vb0dW7ZsETkpXq8XXq8XwWAQ9fX1aGxsRCgUQm9vL2pqaoSRxsMDydPCjWgSLDRjT8Z9MpkU95H3x+FwIBAI6O4Drwwne8a4Z4YbpzzkjpC9Pvz+UduyV4Y/G/IY8qpf3CtjdLxsdPN25J/8WuleyiLOaH8Ovy9yCJd8Tv7iOSh8LHl/yLNGYXy8kppRmJj8XSSBRr/Td4nfM2pLDhuTc7WoDbqPXMTKz4DZOMtwLy+1K39O27l3iI8V9yByD5rcjvx8cM+UnP/Er4lv5+3RZzxcTqFQKBSKcqasRRBfx4N+8lwQCmnjRmkymYTT6dR5h3heCzeSKLGYZkspMZgbtdxoCgaDALateM9D68iwIxFAoTRerxetra06EUTVzQDA7/cLzxPlVHCRwPtNs85WqxV+vx+JREIYS2RUU19TqZQQOZT/sXHjRvT09KCurk7MrlP7ctU6+l2eAefhY8lkEuvXr8fmzZsRDodhs9lQW1uLiooK+Hw+jB07FtXV1bBYLPjkk08wceJE3TolhNVqFd4Kym+h3BIelkX3jBubFLpHzwLflzxm/H5zLxugz9UwM145RiFx8tjIhi2Hi3PeH9kwpX2NthdCNpCNvBfcqDcy1Llw4uPFny9ALyLoe8Pvi9x3yhWqq6vT9YsKEZDope+rLORoTOi7xEUUF1zUJz4e3Dsst8e9i7SvXKij2LEv9F6Gjxf9XeJ/J8yEuFGoq1yohYtZeR/5uaCxplyx4TxvCoVCodh1UYURyhg5JIji3inHxmh9EapWxmeBuZCQZ+EpT0DTNFG2ls9u02fJZBL9/f2orq7G9OnTsddee4nZbzLcyBindVBcLpeI9+cljilZPpvNoqmpCQMDA3A6nSI3B9Cvp8LD0Ci0JZVKid9JGHHvFZ8Zdjqdomx3LBbThbvw47kgoJl2QjbCyBs1fvx4MQ5tbW1IJBIiHDCRSKC7uxterxctLS0i/I/yIHjZZS6MqH/yrD03jOVwHnn2n8KauIeMCyie98DFAB9reeae/6R9aQxlIVXIG0D3NB6PDylyjMLeCiHvT881F59m4orey0VAAH3JcfpO8FC+VCqFdDot8sUoh44LEO415dv4mPJQR4JPSng8Hl0J7GK8M7LXg7crC1b6nIpfyKXxjdoptM2sTzwMjr7Hsmgp9t7LRRLkfXmJ+87OTjQ2NupEljwmyhOkUCgUuwcqHK6M4XH6BMXph8NhrFu3bpBngMfoU7hZb28vYrGYyC3iuSC0H3kiKGyOQtpIRBxwwAFi/aGBgQGsXr0aU6ZMQXV1tchpIaOF8nUsFotuDRp+DWTMezwe9Pb2iu3UJ/kYMpr4uNDvctgUN+4Ih8Mh1tNZs2aNyBmSvWP8fEbhXPw6qagCJeDX1dUhHA6L3JoJEyaIHBsKU6RFQ/nMNxe7mUwG0WhUCEoKeeL9obHgokf2wpCA4+KThyKS4cnFChmLsqeKjwHtSz/pHnNkUSYLIeo/hV/y8TS6D/wnPy/1Sb52/p6vGSN7u4zEjpERzQUxF6W8ZDP3XhqJLN5/PjnBvUG8L7w//F5TkRQqZsCfR/q+ytcnCyq+jyw0SLSRp5SPNxfhBA+z4xhdOxc63HPLvVRciBrdT6PnSRaR/BppzKhNKibBx5i3R8+LQqFQKBTlTlmLIPmfPl/xnsKxNm3aNMiwJmPDarUiEomIYgOAPjmaPC1kZJDBDkB4iGhGuKWlBblcDtFoFD09Pejt7cXmzZuRzWbR0NAgwvK4p8LIMJKNLjoHzQjLBpVsuBrNbBczK61p+aTpXC6HlStXIhQKCUNPNqqKaQuACMurrKxELBYT3jKLxSLWpqHQIjK+5HAnPi70MxaLIZPJiP7JM9NcZMgCTjawzYxGDlWFMzL+zMbGzPCVRYv8kxu1RqJTvg4zjM7Pr8dolp/3s1B7snHM74FRwQ4AumfezCMhXyvfR87xIvj9pXA6ozwqM+FoJFyHeh7k/pgh993ou2vUJy7iaZt8z7iXjY+XUR9of2DwmmXy9RkVbuDP5HBDABUKhUKx66LC4coY7uEBgMrKSlHmuL+/H319fejs7ITP5wOQNwBowUzy7sRiMd06IWREpdNp9PX16UpQ19XViZXjN27ciGg0KsptNzQ0oKWlBbFYDGvXrsVbb72Ft956S3iJKDzOYrGIBSopqZ/PmhvNBpOHisKIjGZpueHCDchChjLfn8oxOxwOPProo5g2bRqCwaAucZyHBMlGOTf4+DYqdOB2u/Hyyy+L4ghVVVXCs5bJZET5X/Kw8XtMuQhkfA0MDAhRmsvlEAgEdDPkfFxkIcRfPHyQVzyTxzedTqOtrQ3Nzc3i/hmJr0KYeRfk+8D7SqIQwCBvEjB4Vt+oXSMPAq+spmmarsCFLMaojWKeI6NjuDfJ7Lr5sYX2kfene2C03g99j7lolIWjkZe0kKDg+8gLtnJkEcbFijwW8rhwwUMTIJS7aNSO3K9CwpKPnVk7PMfL6PksZlJEoVAoFOWBVqJwOCWCRgkyfqh08n777Qe3241NmzZh5cqVWLVqFQ455BAA+TK9mzdvRiaTQSKRQDweF4sz0iudTqO/vx/JZBJ+v18X9jYwMCDKPe+///54+eWXUVVVhaamJlRVVYkiC+PGjcO4ceOQTqfR3d2Nd955By+//LLIKaqpqcHee++N1tZWUQyAjCbufcpms8KTQqFaPEEb0AseubzxcMbQYrGIBUsbGhqgaZoIKeJhREZGHzde5dnp1atX6wzLlpYWVFZWimvy+XyGa6TwayDvTywWE0UjAoGAKDLBc3jIeORijLbz/hn13Wz8crkcEomECB0igcKNbJ5XQfeCnk3ZwDeCf07ClyrwyfleZvehUNuy6OGGrpEnjT4381qaIQspef0a2ZMg329ebVH25phdG3kTyTtI4Xk00QFsy2UxCnfjwpnGSQ4JA6C7p0bXK0O5edQn7nmU+0AvEuJUDIL+tiUSCfGetyMX8QCgK3lO10SVFfmzzo+jVyAQMFzPiY83F7YKhUKhUJQrZS2C+OwuGYl8kUSn0wm/3y/ygMggkr0alEOQzWbFejpkiMiFF7q6uhCJRMRaM7RgKHklqPQyFyV77bUXPB4PwuGwWGR1w4YNiEajSCQSmDRpErxer05AkMFGRRJ49TPedzlMSJ6JNwvt4QYVGdp9fX0Ih8OYNWuWKCdN48U9CXJ4DBlulC9Ai5Jms1msWLECPp9PlL+urKxEMBgUosuowII8k04iiLw/NN68Shf1L5lM6sId+TXzPC/5esgrxc/LF8CkimXcYyd7LrgxK4chyffATBRx4UHPE4k82WPDjVczY9zoGeGVwczC6ug6C4kRIy8TF6Ekeug7xsfezBs0XHHHr4nCXqlt7j2Tr5XuL/WP7hMXRDTGsrAn76CcW8N/Gk0KDCWAZK8ZTcpQOCwXUnw8+We8PX5PeLvy88L3p+qZstCTx1yhUCgU5Y8GoBR/1sv1P0NZiyA5dIWMAb4AZGNjoyiWwI8hrFYrEokEkskkkskkvF4v3G63MBL4TDgVPdC0fInampoauN1u+P1+UXabkqZJGPj9fkyePBnV1dWiCAPlKkUiEbS1tcHr9aK2tlbk5FD/zAwb2fg0GheOkQHLCzKQsRgOh9HR0YGDDjpIeMFoDKhdWQSRh0zTNLHuUU9PDzZu3Ih4PI5169aJNYcqKysRCoXg8XjEwrP8eo3gYoJEp9vtFgUsaG0gPmPOxY7Z9XOPjVElNBojaquqqkpsM5qBlw1FubqZWT+M7hOv0mZmtMvHmbXPPaX0npd5p7bltgB9VbuhcpD4WNKLV2eUv3dGhjX/Hg/leeJt0TNKXlsSqTSBwAUJf37l/DBZFHDBIJ+bv2SxzftWTP6MfD7aRkUe+DPBxSufLDEaL/nvBTC4Uhxto/3S6bShd0nuq0KhUCjKnxwssKAE1eFK0MZoUNYiiGZ/U6nUoFn5zZs3w2az4ZRTTkFfX58om93R0SHCqqg0cH9/P+LxOJxOpwizslgs6O3thcfjEd4HSuoPBAKYNGkSpk+fLgzxdDotvEJOpxM9PT26WffGxkbU1dUJb0YqlUIsFsOWLVvw0EMPYerUqZg4caIIFwO2hbWQKKPFT41i9o0MY/lzbrxwA8tqzZdjXr16NVauXIkDDzxQjC0Z38C2AgFU5IHyS5YvXy4WdR03bhxcLhdqamrQ1taGQw89FH6/H4FAQFcgIhaL6WbZeT/JWOX3uampaZAxR2PQ1tYGp9MJTdOEJ6gQsrEbj8fFArt8xl5eDJYLAqOFYmUvh7yd7ulQIpa8PrR+k1zlTB433mejayXhxseYxpALFd5/XgWMBDM9D2ZjQucjccs9FrI30ajyHICCgsII+dxUNAMAPB6PGMdcLgeXyyUW0yWPEb9HsteX9uF5VOSdpe+kLBqNrqkQvA25CpymaWKtMcoFpDGiyo1GY0QTBBwuaOk7z0UVv++8ip98bygsT6FQKBSKcqesRRA3xHiIEgBMmTIFyWRShFG5XC5UVlZin332wYYNG2CxWOByuTAwMICOjg6xhglVLXO5XKitrYXdbkcikUAsFhMLewYCAdTW1iISiQgjYs2aNbDb7QgGg6ipqdEl9wPbws8ACPFFBRvOPvtsLFu2DE899RTq6upw8MEHo6mpCZWVlejr68PHH3+MWCwGv9+PI444QqzFAuiNSzmkhhu1RsKAG13hcBiapiEUCmHMmDE6Y4znv9hsNmzevFkUhaDPGxoa4PF4EAgEUFVVBb/fjxkzZgiDiQx3Mop5qKGR90L+nRvOtNYSbaupqYHFYhFFKvjsuNEK97IXweFwiBBGEgxkIPIyz7L4NBKacrI9GZzyvSkE9V/2ChoVR5CvRR4/3k8ygOme8GIQsleD+muUDySfUxaNJK6cTqcQRHS/ebEHOb9JFo/yuYzgY01eW3o+ent7RWEOnlfFvWry8TQmcul1uh8kCLmooHyb4WDmFePw76/8HZc9mEN5G428TVx80T6pVEp8B0jwyeGTKidIMWpcB+AXAJKj3RGFYvdAVYcrYyiXhaBwjo6ODjGjSTPvvOpYY2MjAAijLBQKAQCCwaAI6QoGgwgGg8JrlEwm4fP5hAgKhUKDQmtosc9oNIoJEybowk54iBY3pt1uN5qamjBlyhR4PB64XC6sX79e592ga+vs7MT777+P6upqhEIh1NTU6MLa+Hl45bBcLidi/cng5KFAJPS8Xq8YGx5CRMdEo1GsXr0a3d3dwkOlaRr8fj+8Xi9CoRBsNht8Ph/cbreYeSfDSQ4/4ueRDV8jQ5gb6xxeMEDep1D4Dok6fl+4EclFjTxjTt4q2QtT6Drkz82QDV5e5EH2vBQyoM36IX/GBbtRuJSZV8ZoXLmnSx4TPglA4aMAdAusAjAUx/wzMyOf58sB24Q2X3xVFqT0Ox8Dnk9mNF70rPBQWbPnT+7jUIKHH0t/U+LxOMLhMEKhkM4bJ7dtdh5+P4zyyvh+NNlCzwEX3lTowUiMKxQ7hdsA3AMlghSKEpHTLLCoxVLLE5q15bPmqVQKmzdvFqWyySgjQyiTyWDMmDFIp9MiDKqhoQGBQADBYBBerxeBQAA1NTVoaWlBe3s7crmcmF222+2ixDNV78rl8mWao9EoOjo6sGnTJrS0tOgWiJSrl9FsPM2MH3HEEchms4jFYvjjH/+IdevWIRwOo6GhAYcddhiCwSDWrFmD119/HRMmTMD48ePF4qZk5JBhqWmaMCwpNyIcDqO6unpQGAyF1qRSKVRUVMDtdouKedyIzGQy6O/vx7///W8kEgmMHTsWVVVVyOVyouBBY2OjKEEOQKy/RP2je8WNajm8St5OcCNYziMiQ5buM5+xB4zX1OEiz+ic/HxyqBQdJ4stLjqNRJGR+OOf8XNzscO9UUZ9NTJoZfj+fAFPGjPZAOfXbFSswMwQ55MOPIeJPK9UhEQWQfz8sjeD5y/xQhx83Ojc5L3I5XLw+/3iOui+cO8nMHjRURK4srdPvp80bi6XSxduyAUc9yLxMTMSLPJEBr0SiQQikQja29sxZswYQw+wXFBB7ic9j/x7IV8X7RuNRsX48fGh3xOJhLhvCsWosD+A16GEkEKh2G7KWgSRgc6NC4vFgvr6egQCASEqaK0Nguc49Pf3o6amBn6/H8lkEqFQCM3Nzaivr4fD4UBLS4sQUKlUSvwOQOQC5XI5uN1uJBIJAPlchL/97W/YZ5990NDQMMjwIoHBIa9WMBjE+eefj2g0KspCk8do3333xYcffohMJoOOjg589NFH2GuvvRAOh7Fu3TrU19ejubkZTqcTfX19iEaj6O3tFSW/jzzySNTU1IjKauQto+vu6elBMpkcNHOcSCSwevVqtLW1oaamRpS3JmOWQo76+/sHheeRUZfJZERegzzrzs9Fhhr38MlhWWSo0mfc8JPDAc2MUNmTwvN8qF9GXgd+/2lWXhYoHG4M822FoLGg/BUy8EmUGoVB8fGWz2Emjugc/f39qK2tHbSgr5GAM7oWHhpG+UypVEqMFwl9qr5HOXeNjY3QNA39/f26NniVPn4/AAhPBL8PXMRRWKfD4YDH49FdLzf65ZBD/gwZ3UM5HJPWt7JY8mGYDodj0CK/ZveWX5ec+ybfM17xjv6OZDIZeDweXU4ftU3iTBZcXJy63W4hRvkSABaLRVce22q1iskUi8UyKPdSodipjN3682UARwL4DEAYQHS0OqRQlD+aVqLqcGX6b6GsRRCHQjl6e3vhdruFl4jyb8g4IeETiURENTTKYbFYLGhubkZFRYX4589nsuVywXa7XVSV83g8qKmpQTAYRHV1Ndra2hCNRtHd3Y3W1lYkk8bTVtyY1TRNLKTqdrvFYqUUUqZpGsaPH49wOIyenh6kUimk02nU1taiqqoKH3zwAT799FO43W5MnToVbW1togobAKxYsQJ9fX1oaWkR4oz3gdZNInHJQ+fI4KqsrITH40FfXx/6+/ux1157wefzwePxiLA0uha+Ho2ZYW1UoYx/Lhvz3Isley2MSjlzkcMXQyXDmcaWjHd+3bwPsjDk18M9fbIoonMbiSCjUDm+Hz8PhW7K1ynP6NNPEo70O4eeM1ori4s5uvcyRsav7LWgfByqRme15tenIa8QHd/f348f/OAHuO666zBjxgyd14uuiYetUY4OD1OTPTX0Ow/l4uNB4kC+Jv6Mm4XayZ4TLtKo4AI9Q/Kzx58lnlvGx0/+naB77nK5RJEHKv1PAkb2UsleLqN7zyszAhDt8FBaIyGlxM8uiB2AA0AOu7ZnxAFAAzDSmhoeABvY+38gf72XAnhg+7qmUOzJqJygMocbALTQKRU5yGaziEaj8Pl8unC0cDiM7u5u9Pb2ilnVQCAAh8MhiiLwmVc6j+xRIMOBDD6qDOdyuUQcfyqVQmVlpc6ANYIb69QWGX6U8J/JZBAKhYQh5vP5ROlpr9eLt99+G+FwWIT1BYNBXThgX18fgHz4kdfrhdPpFDPNFPIWjUaF4cm9I9yg5cY19ZMvqEoYGWGy54L/zmeg+f5GoVJGBjn33sjeQXlf2kZimSebGyGfj4dnyeKOPx+8v0Z9keFCb6hFOc2M0kLn4+NCIaFUcZDOI48hb5OPnXwuQJ+LRs8WFzQ8v4y8KRTGxtuSBR734PAQNuoD/U5hq/xZMRsjEm5GIZbyPZGP4++dTuegseXCSf7M7LmX7xF/tuT7Y5SDJN87o++JLNbkfvHnmo+JPOmg2EW4EsDPAfwXwL6j25WCPABgBfI5PaXiJOS9QgqFQjFCyloE8bApMn76+vrQ2dkpDIctW7Ygl8vB6/WKfdatW4e+vj6k02mMGzdOeIKoShiFnPAcGzoHiQo5kZtCZMhY8Pv9WLNmDfr6+jAwMIDp06eLsDF5FpvPUpP3ileyIkONFmQNBoPCOKqrq0MoFBJlcadPn44xY8Zg+vTp2GeffUQMv8fjwd///nd0d3cjHA4jEomgqqoKVVVVaGpqgsfjEd60VCqFZDKJeDyOaDSKyspK+P1+2Gw2URSBCkdQZSwutjhGSf1GRrMsHDh0L7nhKoeDcfHAzwVsm4Gnks8keOg+8/WAjM4v94XfKxKCMty45f0fqm0aM8o949cjL9BZrLCifTnktaM1nqqqqnRC3Oh4CiOVPTLc+KZ7Qx4LnrdHlcYqKyvx6KOPIp1Oi7BNXvKZChXwZ4NeRmWk+Xvy6MmCmvaVPUz8fsviil837wOf+OCFF+hYo5wZWQzz77WZsKQy9olEAt3d3boQPvo7xb2tPCxQhucHAhATH3Q9NHbxeFwnhHh/in3WFDuBFwDMA8TSHPsA2AygadR6tPNZirx36Q4A3xvlvigUZYryBJUxXChkMhnE43H09PSgvb0dK1asQCgUwoQJE7B27VqR8xAOh0UOTHV1NaZNm4ZAICBCeCjEbWBgAN3d3eKffyaTwebNmxEIBFBRUYGGhgY4nU6kUilEIhH4fD50dHRgYGAAiUQCra2tOPTQQ+FwOMSsNy0garHkc3z6+/uxYcMGUfUpnU6jra0NLS0tSKVSiEajqK6uFkKnsrJS5CFFo1H09PRgyZIlQuR4PB4ceeSRmDRpksiHoDwPAJg/fz5isRgikQg++OAD1NTUIJPJ4N///jfWrFmDLVu2IB6P4+233xbFILxeL2KxmPCQUdEIqgBHRhc3nOU8nUwmI8QYhUwR8uwzbZMNffrdDBIndN2F9iVDmc5pZozKBiX1hdaeoZwq2pegZ5KLZDMPFoeHDiaTSXR2dqKhoUGsP8VDB2VPA++zHIZFbdO+5IGxWq2orq7WFdHgoiGXy6G7u1t4++i+ycJH9pDy44Ft4VV8odlEIiHESiAQGCSMeagdFy8UVsq9jLxKG5XFt1gs4vmmfSgsj96Tx5dPRvB7yd/HYjEhHACIdrkAomP4sySPKb9nZoKC+kI5dPQ9JMEufzdoX7nAhJE4JDFNXjoaX57HRW1zDx1NPsRiMdPQXsUOwgWgXdrmA8DnmywA6gH0bn1fi5GHnu2KxAGEtv6+DkAFgDOwLSxOoVCMCFUdroyRDZh0Oo10Oo1AIIBwOAwAusRe8tg0NzeL9WyqqqoAQBi2FPJF21wulzC+BgYG4HK5EIvFsHHjRqRSKXR3dyMWiwHIh/h4vV60tLTA6/UK7wkJCkrWHhgYwJYtWwAAra2twnizWCyYMGEC6urqRCgfLe7q8/lQW1uLVCqFeDyOWCwmErKpStWWLVtEYQN+zblcTlR/I8Nv0qRJonpeZWUlOjs70dvbi2w2i87OTjFGXq9XGEdk+MViMVEQQcYoDEcOVSODXJ7lBzBolp3vR23ReWTBIr8IoxA5ObyIjHTZIKZ9uBdA9kbxfeT+GYkevl0WNNwbQeKDniGjnCr5dyOvgtH+5Glwu93Ci0cvLqB4HhwXO+Stkaur8esx6pcsMOR+mgk7jiyWuRCXvYRyno8sTI2ECH9GAYg1hwKBAHw+nwgd5fsPJdTNhI/8XhYvdA08b8tozHh+kFzWWr7v3PMlf494jhfvI/fSqRLZO4AXAbSYfGbBNgFQiB7kCwZ8BOBj5L0kJwFYVYL+7Qr0AfgEgH/r++jWbQqFQjFCdgsRRMYAhd4EAgEA+XV/aM0fv98vvASBQECUxHa73YjFYoOS0MkopnAvnueQy+XQ3t6ORCKB3t5epFIpVFVVibCipqYmsTgqGZsOhwNut1t4ViKRiFiXJ5VKYWBgAEB+DSNaj4jCYHp6ehAOh9Hf3y9maEnw1dfXi1ykDRs2oLOzE/X19fB6vTqPRCaTgd/vh6ZpcLvdmDBhglgkltYd6uvrQyKRQCKREOPGk6TJECdPgpFByfMNuOFLhjxQ2NNjJFBIHMjCwex5MBIB3DMhCwNu7Bp5NTj8mnlfjESNfF1yG7LBSwY7/U45WzwHxIxCoUr8My5IqUIghXHyMaH39PzTNqP8GX5/h/LWmQlD/t0zOo57fWSBy9vgIXR8EkAeCz5RYDZm9JOqTNK6WEYhifw4WUgWun75vpmJ0ULPJX+G5e8JF6nUb/Jmca8YH2f5O8zPYZTnpxgBLgBL2Ps5ALwjbOs1ADcBSCEvEo5ln/0QQA17/xaA60d4nu3hxwBiIzy2GsBjyIvBKdgWAqhQKLYbTVWHK1+48UCiJR6PQ9M0zJo1C42NjWJBUcrbCIfDcLvdImE7Go2iv78fDocDDQ0Nuvj/eDyOzz77DHa7HYFAAJMmTRLFBhKJBPbbbz8kEgnEYjHY7Xb09fVh8+bN+OSTT7DvvvuKKnNkmCQSCfT39+OTTz6B3+9HNBrFiy++iJkzZ6KhoUHk+tC1eb1eHHHEEWhra8PmzZuxcuVKjB07FuFwGNFoFLW1tZg8eTLq6urg9/vR19eHZcuWYdWqVbjkkkvg8/lE4jmFMlGuCQDU1NRgYGAAGzduhN1ux6RJkzBlyhSRh7Bq1Sp0dXXhoIMOEqLKYrGIIhI05tyzww0y7sGg2XMjg00WEkYVtQC9wWlU6Yuuj69HxBeJ5YYxjTEZfbSmEYkOuV3an0RwLBYTOTv8mSF46JeRAJL7Ls/Kc8+mmXCQx8TIsOfila7d7XaLktVyng8XstQeeUmpCiJ5W6jyIl2vfI3yePCCBbwKH4kXWRzyfiUSCWiaJp5p2s7vpc1mE+tUcbHGxQBdL8/hoedJzoHh49jUlE+2kMtEkyigXD+aiJGFp+wd42Lf6LmhMaHcJVq7i+cU0jNCxUzIi80nPrLZrC7fiotCGgvyoDmdTpETxMeeQ20qthMr9GJle+iCvkAA/70KQCWAswHMRV5EJAHcUqJzF8scAB0A1g7zuHHICzl5rG5AXvApFIrtIi+CSpETVILOjAJlLYJCoRB8Ph8ikQgikQjq6uowduxYVFZWorm5GQ6HQ+Tr0D/62tpaYViQgUmGrxxGkkwm0dTUJMKSqHy00+nEuHHjsGHDBgSDQVRVVYkwuXg8LjwyjY2N8Hq9wrNCXpXx48cjFAoByJelzmazWL9+vcgBaGlpERXfurq6EIlE4Pf7MXfuXPT39yMWiyGRSMDtdqOhoQE+nw+apuHYY4/FihUr0N3djcWLF+OQQw7BxIkTUVdXh97eXmEgWiz5HIctW7Ygk8mgubkZqVRKVHmLxWJYv349AKCqqkqMMwkAEpRk8PGZZaMwJzL4+vv7xbok1A9Z4NC+tA/3rPCFTWXjnY7ja8jQveJih4uSbDYr8iLIcJUNWA4XIyQC5OR96p/ZMdyrIRvJ/Nkj4cHPwUUEPc+8HX6sfA+oH1ws0vhRGKSZ58DhcAhhzKsDZjIZEcYp58HwnCnuaePjQtvoeygv2srFCHmkSHBwQ18Wtv39/chms6KSIn92SGzR9XLxw+8VF2Uk4EnwcK8nXScPj+M5b1wAGYkg/pIFBxVlIaFD31E+Rvx5Nfs+cTEj/42jCSI61u1260IgZSKRCKJRtTDLdpMB8NOtv1+FvGdouPwdwH+QD38z48mtPzsAvA1gAoBvIJ9jk0W+qMDO4Bjkq8P9ZYj9zgHQzN43A/gSgDSAX7Dt9yEv/hQKhWI7KGsRBGz7R26xWBAMBtHU1CTCuPjstVlOBVWwIqNGNmi5B4TW6iBDyGaziUIITqcTvb29sNvtmDhxIuLxOFKplM5TVFFRgUAggFAoBL8/H9js8/mQSCTEbDdd08DAgBA7ZHjSwqZOp1OUwSYDyul0ilyiYDCIzZs3Y+PGjSLUjYxpWoGeDCMyqlwuF1wulwjBopA3v98vhAvPATHyXHBko1s29LlByH8aeU1kUWLkGeEGMzcyebvcg8AFl3x+eRZfPodRuJGRAJIxa1c+jq6Bz+rLniazseK/82syyvGgflORgkLtG/VZvidcqHFRxAUPH386B/cEyefkhRFof+7NMLo/vOIf/97z7zd/fukz7pkxGgvZ60nnkceO/87XOKL+kteRe3q4F5LORddP63clk8lBuWg0EUG/0zl5fwHorpsmLii0lY9ZIpHAwMCArs/8uvr7+8VkimI7SAP4/tbfL8XIRNDTAH4NoBHAmch7eP5qsu+zW1/7IV9Q4QgAJyC/4Ci2HpcEcMjW9gCgE/lQu+3lKOTFzIoC+5yKvDXyAwB7bd22EfnwvaeQD/X7vvGhCoVi5KjqcGUMeVFIKASDQdTW1sJqtYr1guifPhloVJ2NZrhpFpRXDCMjiK/WTiWzKYQonU4jFAph7dq12LhxozDixo8fj4MOOgi9vb34+OOP0dbWJsREY2Mj6uvrdWuT0NpEdXV1wmijhVwjkQhCoRACgQBSqRQ2bdqEzs5OuN1u1NfXo6KiAn19fbBarcIb1NraiubmZkybNg1Lly5FZ2cnQqGQMGzC4TBWr16NlpYWEX4XjUZFvpTVmi/FTdXcaL0lbnTy2WS5aIE8202fkVfGKFeC9udtcI8Jb8NMdNGxZFjKoVfk2ePFKkjoARCVw3ilODNBxit0yZ+ZCQmjfYxm7en8FLbIx0G+bqM2jYQWN6r5GNMYUE4c7ctFIjfG+U/uLaJwR6vVqivBzM9H30OaUKBxp/7Sd4u+q3QcfU+MPESapunWBSJjnr5HFArKq9OZhbwZCXZ+jVys0HedvMl0Dvpu8Ovs7+/XlTq32Wzo6ekRnir6W0JeSQqfJQ9bNpsVEyL0t0seG7p+qtwmiz3yMNP9zGQyIiSYwoc1LR/+29fXh66uLt010PjQ3yZeRU5RAt5FvtobALRCn8NjxqcAtmz9fQryIWMNAI7buu3fJsf9B8AC5KurvYS86NgPwHzkCyssAnD01n3/BeAy5D1G/y3mQkxYhHw43IvSdiu2rW30R+THYAWA5Vu3LQVw3XacV6FQDIm29VWKdsqRshZBZLz5/X709vZi06ZN0DQNHR0daG5uRjAYFAY9GWnJZFIsNBoIBNDf3y8M/mw2i8rKSt0/fvKg0DaarXU4HOjt7YXX60V1dTXWrl2LY489FjU1NYjFYvD7/TjooIOQTCbR3t6OgYEBsTZRc3OzzvimWWUyyvx+P7xeL6qqqvDhhx/CbrcjEolg9erVsNvtmD59Ourr65HNZtHW1obOzk6sW7cOY8aMEcLN6/XipJNOgqZp6OrqwiuvvAKn04nGxkZMnjwZtbW1IqxNHqNMJoMJEyYgFoshl8uXM6Z8Azmcyyy8i6C1lSi0S06q5m1wDwUXS3KYj4xswPKZfjL8BgYG0N7ejnHjxonPafadXrKHR+4rbTfKfzGCfybPxJNwoOsiMcHDqjwej7h2CuciYc9zPGTPF/e8GHlcqBw7GfVUMZCLBuoPv1bKI6KJg3g8rusv97xxsUB9SCQSYhIhFouJMUmn0/D7/UgkEsLbwceNng/K3eK5aCSegG35bt3d3WJNpUwmA5/PJxZM5veURC+NH39uaZz4T56HxAUI3R/azhdPpfL7PLSQ8gptNhsCgcAgsUbXRR5sl8uFxsZGVFdXi8kK/nzx+837RN8Bt9sttpO44aKN7oPf74ff78fYsWNRXV0twhT5s8/PrygRR7HffwfgIvY+i7xHRuZryJeHBoBXAXwe+Vyg5chbI04ULpHdB+CArb+vA/B79lnb1p9TtrbXB2DaUBex9XzDCVFzY5vgAfLXeQHy4kuhUCh2AmUtgoBtuTv0j9/lcmHixIkIhULCaGtra0M4HBZGvdvths/ng9frRU9PjwhZ6+vL19t0u90IBAIYO3YsfD4fampqhKcG2DYLm0wmxeKjBx98MJqamuByuYShQTPV1dXVIuwtmUxi9erVorgAeXEAfXiV1Zpfx2TMmDFi1plKU9PaLdXV1WIRWL/fLzw7mUwG4XAYn332GRwOB1wuF/bbbz94vV4EAgFUVlbqDE1uwJHh2tDQgGQyKWakyVhNpVJiP27c0b2g8aH3PA+H552QIc/Dm2h/PmPP+0hjxD0x3GNDeSsej0dXMhjIG8gUFsi9AIlEQhfeRPtTf2TvDjcyqe+0cCUXX5S3IgsREpVcBNHvFIZIwnFgYABer1eXu8Q9Im63W9c+GdJ0bhkjrwYPYeP7GAlaHjpGBj3lbtH56T0JPfKkUn4LhV3KJZ2phLzP59PdezlfJ51O67xMvAAC9TUajSKdTqOqqkp4cOmc3FNEQo17huk+Ub/55Ic8DoA+xM+oKAXPoaJrIFHCvUN83GXhq2n5kEUucmXvE793vJ+8v1zQUC4VfU79JEHMvW382ScRq9gBOLBt7Z8s8qKiDcD4Io79CMBMABuQF0Au5D0taQw9Rdtqsn0u8mFybmwTRsmt7RtFvqwGMAP50LVicG3tG+1/MIZfOEGhUGwXKhyuzOFGsc1mg9vtRl1dnTB40uk0ent70d3djWg0CqvVKsJQotEotmzZgmg0ing8jmg0KhZEpZLXFRUVsFqt4idhsVgQiUSEwDr++ONF+BIZx2RoeDweEQqWTqfR1dUlxBht5wtW8qRsn88nxBvF7ff394vZbWq/oqJClPDl3o90Og2v16ur8MZzfug8NEtMHjFKkKZCCWTEJhIJEYJERhcJPh5SSHCjO5lMCuONDGbZe0D94EYbv9dyTgP3IhmFONHvDocDgUBgkBeLjxcfD9mbRMcYeYhoPx6aJK/rIodx8evmzxRvi+e20E/ujeP7G7XFx40b2bxfXATIhjjdP7N25fwU/izQdiqOQF4oeskJ/nSP+DmpP/I9pTGQrxGAzlNFIpHO53a7hRFvFProdrt15+Btywui8rGQhSIfc3l8Ab2gomPksaQx48KVI48Jh4cQGgkaOT/M6P4Z5fyRCFUiaAfxNwDHbP39KQBfGObxWwAEkBcqA1u37Q3gwxH252XkBVAI+VA5IB9GtxrAGOjFlQXAxK19qCii7VbkBU8OgAdDCzWFQrFj2MPj4cpeBNE/evLukKcjGo3CZrPB5/Ohv78fTU1NCIVCYpFPWpvHYrGgo6MDgUAA9fX1eO6554QgoZLUZIQ2NzfrSiivW7cO//3vf9Hb24sLL7xQrCrPZ6qBbSvaUyEDyvfJ5XIIBoPo7u4W1eD4TG8ikUBfXx82btyI9vZ2aJqGvr4+1NbWQtM0bN68WSx46Xa7RQiQ3W5HdXU1Zs+ejbfffhvr16/HUUcdJWaAyQik2WWbzSYSnh0OB5LJJFwulzCmotGoqCJFY55IJESOQiAQEJ6x/v5+nUFNuUrJZBI9PT0IBAKwWPKlq2lGXJ55NzJySTiRASavoUNjRvkOXCiQoUsVxszC1LjnhueOcOFEz0IymdR5AIBtZaAphIm28eeU4AJLLvdNuFwu+Hw+MTbce0L5ZPxa+P3hAol7JGi8uOfC5XLpqtyR2OKeCDKIZY8C305FRoC8oc89tIB5eWXuVaF7zftHwpPEguyhoDZkAUeimyYH6P7QPfJ6vaK9aDSqE8D8uuQ8Irp/5HmkcSbk/skC1Qz+3JHXhTxnFBopC3KzseTika7dSHjzz/jEBE1E8PWfLBaLCKNUbAdjkRcmXCx8gnz42faSRt6jFN/6810ApwF4fjvaDG9tC8h7qIhzkK8+dyGA327dFtjaByB/fUZrA81BPpQvunWfMjWeFApF+TNsEfSPf/wDP/vZz7B8+XK0tbXh6aefxhlnnCE+1zQNCxcuxG9/+1uEw2EcdthhuPfeezF58mSxT09PD6644go899xzsFqtWLBgAX71q1+JimnFwo0UyiXwer0iZIa8Js3NzaiqqhLGejabxaZNm7By5UphXFAbc+fOxdixYxEIBPDxxx/D5XIhGo1i+fLlePrppzFp0iTU1dWhoqICmzdvxuzZs1FRUYF7770XVqsVe++9N4477rhBRijl6VCoXUdHB1KpFHp6elBfX4++vj709PSIECMKsfrggw9ERTcyivr7+9He3o6amhq4XC4R6ufz+eD3+0WVt1QqhUgkgo6ODmzatAn19fXCK5XJZHSiIJVKwev1itwJqkTX39+PYDAokrapQp7f7xdGGBlu5EHjRjLlb5BBHAgE4PF4xIK2NEa8TDmJOQ73JMgiiQQChRCSCJX7R/vKM+JyKBt9ZuQJIo9RKpWCx+PRzdJzw5fKgXs8nkFeIm5gc0+mHIaWTCZFrpbL5dL1g4fHcdHD+yEvisnFhDy2HJ4zwj/joV+8n/K94J4d/p6EhsfjKZjbwtshEc49KnLpbWqDRFYqlUI6nRb5cQDEey6cBwYGdHk/skDj4oxEBe8v9+DS853L5YSHVr6mYuDPLFVz5EVe+D03yq/j/aPf6Z4ZiUdAvyAu/w7yUtk0RryPimHwHWyrbmZFXix0s88rkPemXAHgYeRDxPZFvniBEQsALJO21WBbBTb6z34kgGsBPDSMvt4E4G6pb5+x9yHkQ+X+ibwo+j2AP239LIB8jhGQr+6mAQiyYz+PfN5TG4B9oBdVCoVi51OicDjsKeFw0WgUs2bNwle/+lWcddZZgz6//fbb8T//8z948MEHMX78eNx4442YP38+PvroIxFu8qUvfQltbW1YunQp0uk0LrzwQlxyySV45JFHhtUXPkNJhqLf79fN3pOYIOPYYrGgu7sbuVwOtbW1qKqqEnlBqVQKNTU1qK6uRjAYxOTJk5HL5YTAoBC7trY2dHR0iNwOv9+PKVOm4P333xfFD8iwACC8FNzTEQqFxEKtVGKbZmFpkUwyRMg4oz7QOPb396OmpkYshuj3+3VhdatWrUJ3dzdisRiWL1+OI444QrfYIv3k+TiUy0FjmEqlEI/HUVFRYRiCxpPLeY4Fvz8UpshzI2h/LlC4AWjkOeHFAOQQH349PMSNRBgdR3CD1mzG38iwJ08bX8eI5whRyFBvb68u94X3j4+B0TNNbXGxYRTyZoaRl4T3gRvFZOAbCSMjo1/ujyxE6Byy2JLzx/iCtnxsSBBTaCI9h7IoM7rvXDhxLyBtSyaTOkOejwEXWfz6uViVx5Pn5HAPIXk/5TGR25LHnIfm8T7R94c/n1x8y881n4ChsUgmk4O8jnx86Z5wscWfDeqXygkaIW7kFy7l0PsDsS18rQ35QgTzAdyLfF4QFU74ANtyhh5E3pPCsQHwA5jFtn0G4ErkK7R9uci+3oh8VTjCatD3CLZ5fFLYltfTh3yJ6w+QX+A0jnzlt32RF3gp5MPpPo+8h0mhUIwqmpZ/laKdcmTYIujEE0/EiSeeaPiZpmm48847ccMNN+D0008HAPz+979HfX09nnnmGZx99tn4+OOP8cILL+Dtt9/GAQfky9PcddddOOmkk/Dzn/9crMxeDLKx43Q64Xa7RdUqTdNEiA4ZCmSg2u12tLS0oLq6GkDeO0XlbF0uF7xeL8aNG6db66e1tRUrVqxAZ2cnBgYGRJtOpxNz5szBmjVrxOruNHNPhgeVqSXjhcLCotEoYrEYgsEgnE4nksmk2EbrC9G1UN8ohKe/v18IP/I60Ex4IpHAJ598gp6eHmQyGaxcuRL77ruvSLqX8yGo0h6FWdlsNpEnFY1G4ff7xb5c5FBSO91/2dCnsZdDuWRDn4fYyJ4b2QilcxmFIHGPDk/85+0YtWcmVOQQKZqhp8pZ/Dz0eyaTQV9fHzweD4LBoDBkZe+WfF55bPhYUIiaGfJ1ySKGG+Ekgig0jAs1wihvpFD/+TjQ8fyclJdDRjiFXnFhy0MRqYCBGfwzOe9GFrUEfQdpnEgg8+fQqLiBmVjhwpd7jXgfueDhIkYeO74PLw5C3x0KT+V94uenvyvy95NeVBCEQkX5RAGVNadQNxoTnjdF51UiaDv5GMDV0rb/QO8R+QKAw5FfPyiJfKgcoA8bawFwF4AlW99PAPAjAKew/YmNAH6M4XmDTgLwra2/DyC/phCQX2vICeB/AHwPeW8QJ4d8kYYTkS+rnUN+cdYQgPOQX/h0HfJCSKFQKEaZkuYErVmzBu3t7Zg3b57YVlFRgYMPPhhvvPEGzj77bLzxxhsIhUJCAAHAvHnzYLVa8eabb+LMM88c1C5VKSMikQgA/Yx/LpdfVLC3txcWi0UUE7BY8gUMaNvGjRsRDAbh8/ng8XiQTCbF+hpkcHDjyev1iupSkUgE+++/P4C8wdDX14fq6mrU1NTA5/Nh6tSpSKfT+PTTTzFr1qxB3gM5/8Xn82Hs2LFIJpMIh8PIZDKoqKjAZ599JhZb/eCDD4SRQ/lGZIDX1dVhy5YtYpz/+c9/IplMoqurCx9++CFqamrgcDjQ0NCAE088EalUCv/3f/+H119/HV1dXSIsraGhAaFQCDU1NaisrITVakVlZSXi8Tg6OzsxdepUIeq4mONJ+nxGWfY8cOOJRFomkxE5Utxgl48l5PwU/h7Q52jwBV7NPB38WLMwIfkz/jsZ/BbLtqpiXGyRIKc8HRm6btkbwtegiUQiqKmpEd49eSbfCG4c8/A4Pg6yIDMSDrKhbuaF4kY3iWv5c4Ln4NBPHmpG56ZCHFykxeNx2Gw24enlfeLnoAIY6XRaV4CDwgdpf6vVKr5HJNDk50Qufy0/C8lkEgMDA2hsbBRV6OTiG1z40bXJ26g96jONKYlBXiBDvlYSOPzZowkFOZSNJi3k3DjyaPNCJVy08vE2y+tSFMHbAH6CfAGEQnyKvJfk71vf2wH8CnmPzLXYlmfTBMCL/KKp+yBfbe2Mra/h8C3oRdiZyHt6vgmgfuvn1OcrkPc4fQ/ADQB+wfrJ4df45tafPch7j+YDuA15j5NCoRhVVHW4EtLe3g4AqK+v122vr68Xn7W3t6Ourk7fCbsdVVVVYh+ZRYsW4ZZbbhm0nS/QSJ6QYDAojCVu9FAewdixY0XeDRkZsVgM4XBY5OhQiWUACIVCwjirqakR1ZNSqZTIJxgYGBDbPvvsM7z22mvo7u7GmDFjUFlZCZ/PJ6qrUb95CI3NZkNfXx/6+vqwdu1aaFq+CltPT49YP8VisQijmrxbfX19+M9//gO32y3WRyFDbezYsRgzZgxaW1sxZswY0fdAIIDa2lr85S9/ETPoW7Zsgd1ux9ixY8UaIeTtstlsIm+KhyUZzXLLYWxyCBGNOeX2cBHAZ/aNvDS8chtP+uYGNAks3o6RYOLt0k/uSRoKHmbEPR+8fw0NDbBaraKkOC+gwM9t5uEy6gf3FJmJEvn6ZMEAQBj/Rp4Po3NyDxvdQzkEi8aZeyH4Z8A2L4nsPeQ5S1xk8t/5NfFQO9k7Qp44+n7xsC6j+y/nOPFrop9cqPNnxW63i+qNRkKJf0/4xAHPSzIbby5MAYhFTek9nxSSvYA89wzYthAwHyM+gUTV9Mj7ZjQudKxRBUhFEbyJfDW0p4rY9z8G21LIe1/uQT7EDFt/vwL5kLl1yAuibxoc+wyGLj19CfJV2oB86Nt7W/v8deTF1be2nv83W/fxAGhG3tNTLG8jv2Dq8Si+jLZCodixaJbS5PMMs42h6gtccMEFePDBB3XHzJ8/Hy+88IJ4X4r6AmVRHe66667DNddcI95HIhE0NzcP2o+8JTzuX9M0USjAZrOJWXVg2ww45eOQ8PD5fHC5XAiFQmI/HjJERgCFmJCY8nq98Hg8yGQy+M9//iMMDQp346E/PJ6fCg5QPD7NxNJMMgkHn8+HUCiEXC6/1kxnZyc8Ho/Id6qrqxOlrR0OB2pra4UAovOQANlnn30QjUaRSCQQjUZ1a+uQh4a8VbyymCxYzIxH/jkXf9y4kw3LQsJAvs/8M8ov4ZXIyAiW2+HGsBx6VuicZkaxkZeI8tP4s2I2NoU+kyuhySLIzLsljze1ZzS+3GNE+/F7Ief20P5kDNO6MnIf5GPoPsn70Plkw1oWE1w4y/3n72UxzkM+6Xr4tfIxkcUL/5x7WvjxNKHBnyuj66f3RgUdzK6F99VqzRdfoL9v8qQAjQ15ROWiFPx6jSYnuNCV+0RwL65imBh5S4olg8EhdEQLgOOQ9x4lMdjLdDTyguYBFBZChyNfAOEgAD8H8BaA8wFcv/XzXwBYg3z4XW7r+5HSB+D17TheoVCUPUPVFwCAE044AQ888IB4z3PtgdLUFyipCGpoaAAAdHR0oLGxUWzv6OjAvvvuK/bp7NQvgZ3JZNDT0yOOl6EysTJ89XM+00kGMRkKZJRSkQSOvIhmW1ubEEvNzc2D4uwpTC2dTqOjo0OIqP7+fjQ2Nopy1c8++yxCoZAwMA466CARGkVheGTk0BomtbW1iMfjWLNmjVj7J5vNIhgMCm/PpEmTAOSFYDgcxv777y9Wpa+oqEB1dTUcDodudpqMOjLk/H4/TjnlFKxevRpdXV0i98hqtSIcDqO6ulokkQeDQSGqyKPCF0wlZMOJG6F0nUYGoCwiCnmS5N/JaKNQKZ43Qwu7ulwunTji55a9FdzINhJBcpiZfO3Ud14AwswLZQR/zki08nGRq4Nxjwz3ApDHTp7NNxp7ygmRBSGdg1eu4yFsqVQK0WgUNTU1OgOcjuVClb5D/LmRi2gYiQyCV6ujc1B/ZZFG3lJZOBE8B4jfGx7eCGwL7aN9ZK8fvzb+XAz17Mpjxe8FjS2NA3mPrVYrPB4P+vv7YbVaRQVHXlKcr9WVTCbh8Xh098voGaBJI/73k3uReAilpuXD9eLxuM4LpRgGFciLluGQxuAcH2ITgCeQL6bwUwBvADhd2ucp5MPjIsgLsX4Yi6GvbP15J/JWwVeRrxJH2JDPBzoAg704PVv7UizTAdwPYNwwjlEoFDsEbZQKIxSqL0C4XC5TXVCq+gIlFUHjx49HQ0MDXnrpJSF6IpEI3nzzTVx66aUAgDlz5iAcDmP58uWYPXs2AODll19GLpfDwQcfPKzz8VAXqmxEYsbn8wHYZgzLIU9kpLhcLlRV5UvfOJ1OdHd3Y8WKFejo6MC4ceNEQjIZQOR1yWQywvNCBQusViuamprQ3NwsSlqvX78era2tGBgYEPlF5N2hPpOhk8vl0NHRgVWrVsHj8QhBEwwGEY/HsXHjRjQ1NaGiokIIo2AwiLq6OgQCAcTjcWEoURliGiMaL2DbOj9kKOZyOXR2diKVSsFms6G9vR0NDQ3Cc0QFF8hwNRKThTDz8MjbhwrxkpPuaRvlmpABSPc9Ho/rQtHkRHmzsshGs+dytTEjUVbMNRiNg7yNDFB6ttPpNPr7+0VSOz+PLCRpTIy8FvwYniNChUTkHCLefjqdRjKZhNfrHZTbRefk6xZxTydNSnCBAkBnZHPPE4U0knFvtI+ZF4yKeVB1Re4Noja5t0P2svHxoueF+tnf349UKoXq6mpRcS2ZTIpy77wfRqLbSOjxZ597qOk6rVar8ErzBZF5CWv+XNtsNgSDQd26QLKXi4ftkQim7RRqS/lJ9HeJ+iJ/hxTDYC6A+6RtNciXxu5BvrKbnELYDmDm1t+7pM9+BODbyOcL/RfAMQbnpAnW/0HeI/Qi8uv7aNCX6CauAvAz5HOPjHjHYNvDyFeT6zM5RiaNfM5TtUkfFArFzkMDSrlYKuXsE2ZOjGJYtmwZ6urqUFlZiblz5+KHP/yhKGY2kvoCRgz7P9rAwADeffddvPvuuwDyxRDeffddrF+/HhaLBVdddRV++MMf4tlnn8X777+Pr3zlK2hqahKxftOnT8cJJ5yAiy++GG+99RZee+01XH755Tj77LOHVRkO0M+8c48Nr4pkVmGKz3SHQiHU19ejtrYW48aNQ0tLCyorK7Fp0yZs2rRJlJnms+JkRAJ6o5PyBPbee2+Rn1RdXY0VK1Zg1apVaGtrw+rVq7Fx40Z0dnYiHA6js7MTr7/+Op5//nn897//xZYtW5BMJuF0OhEKhUR53GQyiQ8//BBvv/023n33XSSTSXz00UdYvXo1+vr6RK4FNza5QUWeMVqHqLKyEtXV1fB4PGIdJfJA0XpCNMNMY2bkJRnqHhFGXhZ6b2S4F9OufK/pGXA6nfB6vYMqn5EBzwUhvza5zzz0i3sbja6JhxOZjRHfT+43N0rpXvK+JpNJITzkkDzZg2MW2sQ9RtzLQN8bIw8GPddUmp08qxUVFaJdGifZW0XXQ55FoxA/ecxJ/HBvFu9/oWeEJj3oWecFM3hSv1z1TB5P+VkkMUIL/JKXSM4nku+JfK2yADN60Vjy3+l6+PNLIkwOjZND+bjA4hXgqC/0t8GobDl/cY/RrsY//vEPnHrqqWhqaoLFYsEzzzyj+1zTNNx0001obGyEx+PBvHnz8Omnn+r26enpwZe+9CUEg0GEQiFcdNFFGBgYQMl4GkDd1tcY5BdNTSLvWTkQwKsYvG5OPfJelk7ky2w7pBc54y3YtqCpEVnkw+rmbW3rffZZoeOK4UvI5x0VgwbgX8ivc/TpEPsqFIqyo7m5GRUVFeK1aNGiEbVzwgkn4Pe//z1eeukl/PSnP8Wrr76KE088UfwPH0l9ASOG7Ql65513cMwx26acKFfn/PPPx+LFi/Hd734X0WgUl1xyCcLhMA4//HC88MILwoACgIcffhiXX345jj32WJHM9D//8z/D7QoAfcKuLIJkeDgNfy+HjtBMck9PD5xOJ+LxONLpNAKBgG69G250c2PO4XBgwoQJYv2hYDCItWvXIh6PI5FIIBQKiXMCwObNm7Fq1Sp0dXWhtrZWGIG0wCqFK2UyGWzatEkYr36/H2vXroXVakV1dTVcLhcqKyt1awXJBhgPR/J6vchkMiKPyOPxiAp03EiUw3/4+I3kfsmiSv59JO1xMQFAV9lK9pqYGeDcwyCLLCNhI7dNGI2LmSdShoxc8rDwAhq8oACJA1n8yNdZSIyRcczzR4zCpuh8ZATziQCze0nw4gT83Hw/ud/ymj1GY2mGfK/42FEfScCYedOMvDLA4GdKFgv82vjfBfnZNvJCceT3ssCk/tLaY1QFzyh0T4a+K/zaeJty3h7tx71iu2JhhF1p/bohGQtg/dbf7dhWXOB45BcSvYjt24Z8+esEtlWFM2IfABsAGEeO5HOKViNfVvtDbPMuAfkQuRbkxdGO5hHkc4tewuC1hxQKxU6n1NXhNmzYgGBw2wrJI/UCnX322eL3mTNnYp999sHEiROxbNkyHHvssdvXWcawRdDRRx9d0Fi1WCy49dZbceutt5ruU1VVVZJ/LNwo4JWN+EwoN2pILMn/5KlCW2VlJTweDyKRCPr7+9Hf349kMon+/n5s2LABdXV1aGxshN/vFyKBz27zmf1x48Zh1apV6O3thcPhwN57741Vq1bhs88+w+mnny7ETU9PD5YuXYrx48fjyCOPxD777INVq1aJkt0NDQ347LPPMDAwIMJygLxBFgqF8PHHH6OyshKJRAJbtmyB1+uF2+0WYWE8pp+u02KxiNnxdDot1ieivnd0dGDMmDG6tY6MjG1ZaBqFABmF/vDPZPFgZLjL5+OJ3HKRC77AJIVhcYyqhMnwPlO73FNlNBZGpauNzlNImNDn5L2IRCIIBoNixp/nsJDYNmuDG7tGYkm+Fvl6uIiQhQO/BhJQcrEKHlpHYVaxWEx3jNFx1F+OXCBArsLGodLO9F2n42UxxAUDP7c8jlxk0LnJ85LJZJBMJk1FmZGwKtabysdCLrpA18lD1GhtMp4bxPtL480FL/dM0b40ycOfBS6AyGO0q7ErrV9XFHHk187hf0L/DuAx5Bc4/RHbnkQ+TC6GfFns/wMwlX3+LIDPAahFXtDoozO3cQ/y+UDPAti8tS1s3X8N8msD3YThl9hWKBTlTQmd+8FgUCeCSsWECRNQU1ODVatW4dhjjx1RfQEjyqI6nBkU1kVhQzz/h/7xZzIZ9Pf3A4DwdMjGOw/xICOAqrmRFyiVSmHdunVYt24dBgYG0N3djf32208UEaiurhalsjVNw5tvvomamhq0tLQIg7yhoQEVFRXo6enBli1bEIlE0N7ejs997nOoqqqC3++H1+vF9OnTRTupVAozZsxAS0sLJkyYgM7OTuGtsVqt+PTTT+H1elFdXY1ly5bBarVi3LhxqKurE9dE+Ruall9gtbu7WyzqmkqlREEBWtuGiloMte5LIWg/bjDxYhDcMJRn4mXxRBjlBPFZbcpX4tUBeWgYGY1c2Bmdh3/Gcyt4qBDPbZH7Xmg8ZAEu57dQ+J7H40FNTQ0qKiqEmEskEroiFXKOiVGIHxcbFAbFy1NzbyafFODH8QkEEtJckJl51/jY0/0hLyfvJzf45edDbpPEDDfQyYi32Wxi7R5eBpzEPOWIZTIZDAwMiJww8rJS+CQX0vQdTKfTIjSVQk9JHHIxTgKkp6cHsVgMzc3NQnjI4bPy80F5PkafUb4e3Xe5EiLlY9F10/PAQ+toHPkkAl0D/T2hv3dydUV6TgYGBsquMMKOWr8OMF/DriCbkfe8pKXtn0fe43PV1vfvAqAJzzSARuRzaDTkRc+rWz9Lbf08g8E5RZwc8tXj6pAXYCSE0sh7kp5Bvjpc4aXIzBmHfFW5QjyB/KKtY5EXZONGeC6FQrFHsXHjRnR3dwv7tFT1BcpaBPHZeW7MAfkqbvJn8synDO3P8xHoPOQ5IePD4XBgy5YtiMfjiEQiIvGbDIwtW7YgFoshFAoJr0owGITH40FnZycikQjsdjuamppEVTlKdJYXMySDm6pr0VpBFosF++23H9xuN9atWydyeiwWixBqVK2Orpk8RnJitdfrhd/vFyF/JFh47lOxAojgYsIoJEg2vuVjuRFcKASK7p2ZB8ks7MjISyWHEBp5TbiH0ah/fD/5mgr1lZ5VnhMDbAtFk3NRjK7R6NrlBHk+9txrQ/sbjbfRNZl5+YzgRnih42RPibyv7LmS+0pCihfy4MY+wYWtLOL4Z7QtmUwiGo3qilXQMVxwUb8cDocIeaUJGe7RkUNN+d8n2SPHn00udOWiH0YiXG6fjx0fM/Jsyc8I7xdgXHp+V2dHrV8HmK9hV5AcjAsCyIUFpgB4HPk8HmBbYYTTkRcwRulKduSLJOyPwflFQF7wdAHoRd7z81/kK771IO8RGm7kyv8gH8IX3dqn05CvUmfGOciH/dmQLwqhUChGldFaLHVgYACrVq0S76m+QFVVFaqqqnDLLbdgwYIFaGhowOrVq/Hd734XkyZNwvz58wHo6wv8+te/RjqdHlF9gbIWQbKhQtuA/CroAERZavqMh0fxJGJgcAnnXC4Hh8MhcnIAIBAIiFdnZyf6+vqEZ6WhoQF+v18c8+mnn4pqci0tLfD7/chms1i9ejXa29vR1NSEqVOnDkr85tdBooqv/UNtUnnv9evX47PPPsPYsWNRWVkJh8MhFl6l4gdULS8ej6OzsxPZbBZVVVXwer0iDIauS57l5+NrZuAPdZ8IM6+B0f5mRpvZWNF28vhwI5l+yt4S+TxybhkAQwOTe0tkw7OQiDDyGhl5buhZ5WXgKdyJi38zL5A8Hryf8jaeJG8mSKlto7V6CCNRwuHnMQrxkr1H8nfb6HzydRqF7JF4IC8wbeeFDbhnRc6psVgsIjyRcvq40Eqn00Ko0t8kWqzZbrcjHs+vbknfZZ6vJz8HsgCjayMPFvdSGeXvyPfNSPjKY03XTM+aWcEIuofDnQzZnSl2DbsR4UW+nLTMSpP9w8gXHHgKwJ+R9xhdhm35R5ws8gLoeuQ9SJz7kS/iUAwrAKxi798BcMrW359EvpjDtdhW5vsY5CvadSK/DpFCoRhdNJQmHG6YbRSqL3Dvvffivffew4MPPohwOIympiYcf/zxuO2223Q5RqWoL1DWIoj+OVPlqlQqhe7ubvj9fng8Hp0hTNXC5NlUiyVfLlqeDeeGUDAYRHV1NRobG6FpGvr6+rBixQq0traiq6sLvb296OjoQHd3N6ZNm4bjjz8es2bNwnvvvYfOzk689dZbQpnabDbsvffeGDt2LFwuF9LptKj+RkYQzTRTHgKVrQXy+RQUAkgGyuTJkzFr1iysXLkSqVQK8XgcPp8P06dPF4ZzJBJBd3c3wuEwMpkMpkyZoqsWp2maqCxHM9uapiGRSAghORIhxA0pCo2j48gbZZRPM1L4rDU34Gh2vqenB263W4RQyufmBnQ4HBaL0QL6GXMyfGVDvRgDkfojCygeepfNZtHX14dAICDO63K5dEa+3Ca/Br6dxp3nVFGJdO5FoM9l7wkvn82fBSNPmyxoeE5RR0cHgsEgvF6vrj/y7wDEs0/hYbJw4Z4wnhtosVh0VeBoYdFcLif+RtBzT3l3vB0uMKhfuVwOwWBQXFM8HhfPT29vL2pqapBIJIRHlY8FPWPycyl7omVvFR1HfweoiAnPeaKwQi4q+Rhw8UMeLJvNJtYF4+F1JLJramoQDAbhcDhEvhP3PPKxLRd21Pp1wPaVfy05KeTzfa5m2y4DsBjAxwb7a8gvfvozALds/VmLfMGCeWy/GIAfFNmHzNY2gbz4sSOf50TOtNXIF3DoB/DCoKMVCsUewtFD1Bf429/klZ8HU4r6AmUtgugfPw9Ro0VRycDhZZLlmVdgmwHB3/OZVVlIAUBlZSWmTp2Kvr4+VFZWIhaLoaurC1VVVfD5fFi/fj3sdjsmTJiAiooKvP/++3jvvfdEGWqHw4FIJAKPx4N4PC7K7vJZWYslX+aZZmhzuRycTicCgYDOyPX5fLBYLEgmk+J3i8UCn88n/jnncjm4XC5UV1ejuroaNpsNPT09Yl8ybCg/iBs53CvBjWQ+XoU8NvJYUmI15TUZGfTyfZHzRHjIkOylIeOU9uWz7rwSHjfu5LwYOi8JSNmzIIsGuk5+//gzxUsYmy2eKj+TdG95kn8ikdBVJ+Tn5J4VXiCExpGLJ1pDh4xgHtpF3gvqBxcGRveI/87XNqJwTRoHLgTkxHruQaKX7I008lzIbZAw8Pv9ImSUjw95bUj4F1NNTfbKUN9ofOla6Hq5kOW5N3SsPHayp4V7pvl4xWIx1NbW6p5dOTeO2uF5U+TF4n/H+LXQeShnkNqlvwH8nmQyGSQSibLLCdrZ69dtN/9BvohBLfKhY5du3f5rDD3bmkN+3aCvI78Ahh3AqVt/8rLYLuQXRAW2rQ1EtsSZAL4F4Oit72MANm79/TcwDrMz4m6DbZ/AfPFXhUIxCli2vkrRTvlR1iLIKBSIDBb6x89nYs1mMLmxTPB8IG6k0fvKykphGNCMaVVVlfDiULKyz+eDx+NBb28v0uk0BgYGxCKr2WwW69atg8ViEUUbyGjlYSuyMSsLEjLuotEoUqkUPB6PMATJ6KEwF/IkdXZ26kQGiRMy2OUwG26EyuMOGHsiZK+AUWgOD5GS113hP2l/I/HAyx/zcCpusHKvIR1XqEgCecbksC3u2RrK62PkIZHHSt6f7jcAnYeQxttIFBiNCW9HNvC5SOTPND/eSNyaCVZ5H94H+fkpdP1G7fD+cNHLka+Pcqrk/soeYH6dRiJIvgb5M96W/NzKfSVhwRe8LWbygNqIxWK6MEi5Dxw5r21gYAAej0d8xotdyOPB25LPRx7QXdETVCi+vKWlRaxfN3nyZFEi22z9uu2JLy8Jf9/6mox83sw9yIuftciLnH8in4NTiLuQX/9nbwBfQD7/hy5jAHnP0D1s/2MA/BV5D83TyBdo8CG/fpGX7fu/KF4EKRSKXZ9RCofbVdgtRBAvAUuzmiRMuCeDYt7NZkR5pSn+GZ/15zPVFRUVYiFITdNQUVGBbDaLSCSCzZs3i/O2tLQgmUyip6cHPT09qKqqwsSJE7Fp0ya8+OKLSCaTmDBhAqqqqgaFz8gz4rlcDqlUCtlsFh6PRxi0mUwGa9euRV9fn8hLampq0uUNUFvUZzLyeYUsvlI8D5WSf6fxlGfKZfi6MrxsL/2eSCSQSqUAQISp0T0yMj45FKpElbNIlJKY5LlBPGyNPGsul2tQ2BA3MCnkkHuiaIxkTwR/nng4GY0XD4WSj5GNWR56Rn2X80jksZGFAfWbjzffh4+zy+XS3UN+vZxC98NisYiwTe6J4JMI3PiW8074+YzGlkI0afJADokjuJinkDpql3//6T0dT2Fz/HvCr4FD31EK16M2qF35/gNAKpXCwMCAyM2Txaw8lrwfFosFfX19hkKdT5rwZ4Q+z2Qy6OjoQG1trSia4vP5dONPolHTtlXQIy8h9zbxhVZ3NXa19etKwqcAvohtVeCWID/ZehLyJa2BfEGDodb3WYh87s8vt75fD+BG6EPkFiKfI0RemjeQ9zwdOMw+W5Ev6KC8PQqFogwoaxHEDSGn0wmPxyOqm3V0dCAajcLhcCAQCMDr9YrZcR4fD5gnk5OBwOFiiHJzHA4HYrEYgsGgOCYUCmHLlv/P3pvHyXXVV+Kn9r2qN3W32los7zueGI+tMIBjPF5wGMCezDjxECD+4YSRyYATIDAM+8QTkvlBSIz5JSG2M4NDQhIgOMTBbDYYGYjA2NjGi7xobbXUW+17/f4ondvn3X7VasmtpaR79KlPV9V77767vdI597vcvZidnUWpVEI4HMbw8DCSySSSySTm5+fRbrexYcMGPP7440in08jlcoZI0vefhFsJVbPZNMQ/EomgVCphdnYWq1atwqpVq8xmqU8//bSHyG7YsMEID6ZZJlKplLEcqRCwBSC/I8kkCVxqjEhiC4WCcenpdLoZ9KamplCr1XDuueeaeusYHIh49yKptnXDPkbCqKTxQKBgostdrzoB/umqdQxtMqvfMSNfrVZDJpPxlKfX+FmG9F6a6MMWDiyDoteux3L7RKEbetISqu1US5/WWUW/nzCg+NH+0f5SsaLWTbaNczWRSJhraZXhODLeirBdzbQuFMd0gz3jjDM85zGLI39jONfK5fIB57SWo+OreysRfmKYdVSLcavVQqVSQTQaxeDgIBqNhhkPYCH+Sp8JlmNbBA/meTmSuOwA/uWBwJHbv+6gMIiusJmDd88gYi+Ac/a/34OuZehrcvx2dPf2aQJYKjv37+9/RdDNHHcPupamXng9gE+gm8UuuL+eQDdeqIZuJjv7JzCIburrx3HoabYdHByOLJwlqH/BlW7NqETSQzHEFV5g4T/1fD6PSCRiMqPZ5I9EivuyqFuIkgB+z/1pSCS42sv9hrg3EPeUaDabmJ6exszMjEms8Pjjj6NQKODss8/G1NQUEokEBgYGkMlkTJ003qXdbmNubs7sI8PAaWaPI8ljIDjgTSQRjUbNnkoDAwPGTU7P6+WSxL5SweJH6oPBoCGEgUDAbDJLkpVKpTA6OmoIsx234OfqpeWzHrphrVoF7bKUgNquQHb5fnVQQUjiqGVpnWzB49eXdpmcM0pE7T17bFGhYszPcqNWTltYqKjQ8+z4Elok7c1ZbUuWfW91u6PlkGVrNja1tvq1gdY9e75pObTK6D5IFPUUIhrgz3b5jadaV9gPfuKZMVCcd1ovnW+MsxoYGPBseMx72osOhFoQmejFtnCyLZwLjUYDs7OzGBwcNHUcGRnxJAPRPrDL0bGy+5ip9o9FEXTMI4CuSFC00E0YEER389PnDlDGGLpWn8H91wTRTXywCcCj6IobltsLlwP4AywtgADg7v0voLun0N7975lp7hIAP7auORndbHHH3l66Dg4OvdAJdF8rUU4fou9FkE2ISZrS6bQhQDMzM+Y/+3a7jUqlYtyh+J29kq5kR0mouoOoGwqJZDC4sM9PLBZDJpMxdZmbmzNBxaVSCeVy2QSpT05OotPpYGJiAtu2bTOJDM4880xDrijoSMamp6eNK0s4HEYikcDQ0JDZ0DGdTqNWq5mgZ2DB/YUCiavM2n82bOJzoJVs+zqSLbrh8Hs78xqvWU75KjBIcG2XRSWkKuL0vW0Z0Xb6Xcfv7XTFNjG03d78RJl9D/aD7YLXyzrC+vtZpXpZDLQtdj+RrPsF2QMLosKvDfZ9+ayo+6nGfy0Vn2P3hwobe260Wi3Ps6cb/PJ50eO2Syfv1Ww2Pftz2f2niQLUwqKizm+c+X0wuJA2254jfiKIv0u8nq5s+ptkzw8mNymXy8jlcuZ6JmPRMbGtWyrqmCDF/n3VueFwkPgAuhnYiAq6MTcHm1iOWxl9HcC/l+/PR3cPoDq6aal74V/2vw4Gc+hakA6E55Z5noODg8Mxgr4WQUrm7NVZzYym1hwASCaTxh3Mz3WKRIYuOCQakUjEpKCu1+sYGRkxrl3AYhJDwTEwMGA2TlXSnEqlcNJJJ+Gxxx7zWIx2796NXbt2oVKp4Ld+67eQy+WMuFNS89xzz5mU4NlsFuPj40bk0RUoGo2aviCh4h5B7XYb8/PzeO6559BsNrFu3TqsXbvW4+Jmxxj0Iu9+41Kv1xGNRk1WKfrf+1lktN8OBF0hJzTmpVdCABVaSh613rRW2CKEY2mXZbfZtm4Qdp38+o5xWTyme7ao5YTX23E+dl/a7/U7Wg9oOeVeNhonBiwk1FCrl+2eye/VqgosWDpYpibx0P27eB3vqWXbcXFaDz5LPM/OWkYLKO9NsWP3hfY/+1DrYqcYDwQCZvFCk4nw/FgsZuYQr9XU1pwndpIT9q2OEd/T1da2YLF99XrdWH0HBwc9Io+p6NkGihzeE+i68FWrVRPzFggEjPDk+DOxC93pHA4SfHy3AzjtJZZ17f7y3omuZYdlR9FNajCIbiKFH6PrpvKu/cevRtfVzcHBwQFAp9N9rUQ5/Yi+FkFKkOgSFQwGTerpWq2GcrmMfD5v/iMPh8MmPojXqVVACRhXvklUKpWK2ReCpJTCxBYI6nZCgjIwMIChoSHMz89jeHjYrN6Ojo6auu7duxcjI92ttPft24cvf/nLSKVSGB4exqmnnop2u202O12zZg2efPJJhEIhZDIZfPe730U8Hsfq1avxb//tv/UIQMYRkUBt377dEKF4PI5SqWRIm71prG0tsN2zlBTrajiJJy1PTFZhB8uzz3oJIZtw26JVr0smkx4rhg0VKVp3uxxbxLAPWfb8/DwGBgZ8V8dty4amHdZ5wXvqqrudnrperxtLn21xUEKs48Jj6vak4DND0kvXp0ajgV27dhmBEggspGnX54N/S6WSJ0kAU5ADi129gsEgCoUC2u22eT41rTRdtdTion3PNqgQ5XhXq1WzSJFKpbBv3z6Uy2UTB8S0+XYihPn5edMuOyGHvQjCZ4PtDQQCJqFHu902yTiYgtt2oVWBows22j/cEJnjrO6HzMqmLnma+ZALHhovpfuJcZxp/VYrH89jPCLnAxNm6DjaSTQcDhL/im5ig/pLLIc69E8B/BW6cUPfRFcMxdDdwHQICy54/wvdTVD/CsAuALMAzj3Ee/8EXdc84gsAbu1xroODw7ENFxPUv1BybLuj0E2MsS+aMY7EjkSiVyyC33s/9ykl0VqWHddBoZFMJo11ptlsYnh42GxKWqlUMD8/b4hvtVo1r3a7jbGxMY9lihaWeDyOTCZjMtM99thjOO2005DL5Qwx5cp1vV7Hrl27DHmKxWLGOkYyqyROBYH9nu0m7P1FSKgAoFarLbLQ2YReBShFh4pT1rlXAgWuYJNQq6sar+e1mkZdY61UbOg1GltVKBQMwdS66Dyxy2m1Wh4hqCKSQptkttXqbpbKviC5ZZybjoVaURQk9ToGtqXKTyiq5Uf7i585tqwnPwNekaKxRp1Ox+zdw+tIyOnSZrt6aX16Wff4jPPejI8LhUJIJBImGYJmbKNlyN6fS+OGdIy0H/Q3IxqNGqHEl1qG2c/aP/rs2M+Y3xjob5kt4HU8/PrGFq56rVqjeJ5axtk/KnJty6nDQeDP0RUu/wEL8TUrgfL+1xy6MUFhAD8EYO/vmt1/rARgNbp7EP0QwL89yPttRldw0ZD7J+hmqvszADcfdO0dHBwcjir6WgQp7NStJD6xWAw7duwwKaEBmA1H1a1ErSC6gq4Eo9cGjvaKrxJ7kp5wOIzZ2Vns3r0b5513nqlrq9UyG7qSCBYKBUPmE4kEyuUy5ufnMT09bUid1icej2N4eBgnnXQSfv7zn2Nqagrf//73EY1GMTExgVwuZ4g73V52795tXOYymQzOOeccJBIJhEIhs9Es3aVU6NkxI+wPQkUJXWiALvnSjWHr9bpJTKEuOWotYeyWbhjKz3aAO9FqtUyq8nq97nFNotggUS2VSiZNdq1W85ynLk6ECp5yuWzKVpcsglZEim8S53K5bCw+HEe1NDB2qt3uJr7QbIGch7bFiu3WMbHTGyuBVStCOBw2CTJCoZAR6CribHc0PiMUgToHVARx/lCIpdNppFIps3fWUunqe5Xp1wfcJJifR0ZG0Ol0jKWObda5pdZJWm80RbvWx7Y80v0smUwik8l4BIYmL+nVJpZpZ1bU3wv9XeHvmI43BZktouzFG7UU2RZKtVLqwg3nRyQS8VjP1BLqcJCYRNc97XChiq6FJgDgOvn+UwDW7X//WiwkTwgDuAjA36Obhnu5yQwuBPBf0U3NDXQzwdUAPIKudeh/A/gvB199BweHowSXGKH/QXKmZMV29yiXyygWi6jX60ilUshkMsjlcoZg1Ot1zM3NGXeUQCDgITh2hjXAm/JX66ICie9HRkaQSqWQz+fxox/9COvWrUM2m0UqlTKuLuFwGCeddJIh8jMzMyiVSjjttNNQq9WwY8cOPP/888hms0gmk6hUKnjZy16GUqmEHTt24JJLLsGrX/1qlEolPP/887j33nuN9YlxRFyJbzQayOVyiMVixiLCnelZFwo19iNX89Viwqx4JMwUUmrlALwxHLRsqEABYIQB+y2RSJh7xWKxReSNsF2WWBaJH6+laxnHi+5YHDfGQtgWPZ1P2rZsNotwOGxINM/TetEqwSQUzIbH4zpn1AWq1WrhzDPPNGRU04fzGu0DtsXPqqB7RXEuq7ilVYP3YP0oYjgHbCGlBJl/NfmAgoJLFxns/lLYgoCwyb+OP/tErX+2FYzH1ILF+EAKDT/4WTtphVJLnIpP7Su/celVvv2bAsDE9Wjfqeuk3ot/7TqpNU2fJV084vMbCATM/mfqzudnNXc4htBBd7NTYhDdtNr/Yf/nf9z/UrTRTdzwOQC7fcqMo7uP0PsA/HcAf4PuhquKF9HNIvejQ6+6g4PDkUeg032tRDn9iL4WQUrM1HUFWCB+AJBOpw2xYBwEXVmKxSIqlQoCgYBZndb/5P3+07dXU9UCoORFr2XAPUkmM9Yx6QHJXavVQiqVQrVaRa1Ww6pVq5DJZFCpVIzViGm2I5EI8vk85ufnUSqVzOpyOp3G+vXr8fKXvxw7duzAvn37EAgEMDbWdeSu1+uYmJgwFpVOpxvMzVVwe5W601nIIKcCD4BnpZ3HbbKkq/1qUWJMRTAYNBYZfrbdu+wVchWjSv7VdUqva7cX9o1R9ye9j03m7XJ4X46lxveo4LXrrWm77TmibdAEHu12G8Vi0Ygo9qe6LylUUGkblupDbZeOnU2o9R5art8eUXY7bTGgsTBqxbJFggpRHTM+Q/b4K5Tw25YRtbzZ1lxNkuLXR/q8c38wfYb0XL9+8IPdDh0DlkULqGa+87MuaZtsMW6Prf5uaf/zPhSMbJvffmUOLwEhAG8F8BeH+T5/CeAqABl0rTZf8DnnNwC8G93Mbjv2f/ckgO/tfx8H8B50s78VAPwqukJrn1XOHIA/XrmqOzg4OBxu9L0IotWCn5VA8MVVbgYe12o1s+rNgGEAxmpAMsANSW3iraTO/o6kt1QqmTKZqpeELpfLeVb86S5Fl5NoNIpUKoVOp4N0Oo1EImGsWbQQFYtFBINBzM7Oolgsmr2HBgcHEQ6HEY/HsXbtWrNZbKPRwODgIMrlMmq1Gk499VSTPIJ9QILZi7Czvn5E0SaYPFevV2HqRxD9BISSX3vlX0mfbdkgSCLtGB/7HjZxtMdUz+XqeS/SacMu137v1ycknn711flIsC69xNFSrlO6f4/e3+4HvQbwuqVp/XoJEBXI6mJn94vfWPSCnxBSEWRfq0k59DoVVrYA8rsnn2vNcuf3W2CPn98c1zJ5vlq8NKujX9103vsJ8aX60O5/Fal+LqdLCTqHA2APgGfQ3asnhm4czV/Cf5PUlcQ7ANwHfwEEAJ/ZX58PyndfxULcTxrdBAt/BuA7ADaiK+KelvNn0XXHU7wawMPouss5ODgcm3CJEfoXjDfRDEi1Wg2JRMKTCY2WDWZBymQyAGCsKdxbY25uDueee65ZeZ+enkY2mzXXqQCggCLxoJtVo9FAqVTCE088gWQyiWQyaWI84vE4EokEzjnnHLOyWi6XMTMzg1wuh0QiYWIwhoeHMTIygmKxaATT0NAQstmscZ9rtVqYnJw0WbC2bNmCc889FwMDA6jX68hkMiZdbq1Ww9zcHF544QWUy2Vccskl2Lp1K/bu3Ytms4nZ2VkkEglkMplF8SQKddWx43H4nQ0SJzszle47Q9ebXmTNTtYALAhOJf6aDhjoWitYNtNBa12V6NmktRdh1dgh+9ylCC/gn+parUlaF1rm6EamFiFb9HAuKjHnfGUf2O5UBGO1VJjbUEsBP9NlUsk/rQi2qNB+s8WqLWD0frYgsl1e2Qd2n2of6T1sa5Z9f72vn0WRdQDgSa6g51KQqxupXwyQn4jxc6XTRQWNBbRBocQ6aDv8LEcqtFTY67PGWEtav2x3RoeDwLfRFQt/g4WsbaejKyYOJ4HYha6gOR3dxAi7rOPPomsFWoeFPYZet/9l42oA3wDwSXj3I/oBgDdZ5/7z/vN/iG7MkoODw7EHFxPUv9AYkna7u08I3dvoakUCyOO0GhB0o+N/+gxG73Q6KJfLZoNDkjsVPhoLQ+tSuVzG5OQkWq0WVq1ahaGhIc8Gp3TBUqI6MjKCcrmMSqVi3NlYbqFQALCwws7EBLRSDQwMYHJyEs8++yySySRGR0eRTqcxOjpq2lCv15FIJDA4OIixsTGcdtpp+Na3vmXic3K5HIrFoiFtdE3rZV04GHcYdVO0CSwD8vm5l/hY6r4sW4PR1bXQdlXS8oPBoEmvbM+LpdqoRFrdkPysIvxru3Cp4NJYIyXv6XQa8Xjcd48bOzZGE1moRYEkVq0Iau2wE35o//TqA1s4ap/o9fY4+omV5VgtbPG1FJj9kIkxGCvlZ3HyqzvbpbAFL9tfKBQwPT1tBCTvwz2X2L9+MT625YdQqzbHmIssOtf5u2Zfo/Wz676UtY1g4pRwOIxUKmViJFXQObwEPALgTAAJdMXIkwDWohuLc7gsQu8E8P+iK7buRzdxgsb0nLf/70/QTXxwILwSXUF3IbpCKA7gEngtQ8QD6FqOHj7oWjs4ODgcdvS1CAIW78lCiwNje5ToaLC8EnymkKZYqtfrCAaDGB0dRTQaNeSSWaGYRKFcLmNwcNBYlihcmHyBe5QowbUJMYmLBuwzY1W1WsXs7KyxDiUSCUxNTZl7JxIJAEAqlcL4+Dj27t2L5557zgTYKwlut9uGnJFcMxaKGcECgQDm5+c9lhOu/tokdymrjd/4ADD18LMyLacsPyJruzbRfYif7dgPv/rZFgeOi591oVcZdl3s9rDddIvUe1IY+11Da6edEY31UXGjK/p229Q6o3vdqJg70Dho27Rv7bb79RmtJhT/aoU70NjrcT93V7uNaqmx54rtKtkLflYg22LE1NW00CnsMVlK7Nvt87uXzhE7sYQd16ZWJLtOnU7HJEKxFxDs50R/I1VoOSG0Aqigm7Sgjm5SgTPQjbkJ7H8BL91NhdPki+hmhgOAf4+uILlg/2edIm0Ai43AXdc3+/0v7f/7cQC/Zx3v7C8riMPv6ufg4PDS4Nzhjg/wP3xmeeJ3DIan1YjiSFfSuRKeSqUALLgDDQ0NeUgoyUGj0cD8/DxmZmYQjUaRzWY9dWA2L1tAKImxV2dpfeJO9JVKBcViEYVCAY1Gw5TFdNmVSsWQmEwmg2w2i5mZGezatQuBQADr1683FoZAIGASK3BDycHBQczMzBj3MWabm5+fR7lcNv3IPtEVbSWgfrDFg47RgYjzgeBnJdCylMzZFiy/gHq1/nDc/dp3oHr2ajPBlXQKQT8LjKaj5jUkrJq10LYiaFttMqsuhNou7QOd25yntmj06wu/OaD1sUWQ7nfUSyz26ltboGo7VFCxX1l/OxZK66z97yfKbQFln8M28T72vfyET6++JFiGHXdnC7heZdjPlF//2tkJ7f60683xU1dMhxVC1Pr85+gmKgCALQAufgllT6KbGc7GOeimxG6gGwtEXORz7hkAnpLPRQAnYSGL3Af2vwDgVHRd6zroutcVAJwNYPuhVd/BweEIwImg/oXG59DVLR6PewQOkyEwvkLJH11YaAHKZrMm4UCn08Hu3btNUgJgcWrc8fFxJBIJk7Wp2Wwim81i1apV+NnPfmZIL8UHy1WXJY0TIJFS3/vZ2Vkjjubn503ShUKhgFqtZlzn6vU6Tj31VGzfvh3tdhsTExPGRQ8A9uzZg7Vr16Ldbhs3nkQigXg8bvYlogWKxC6VSiGZTC6KYzkY2BstagpjO/7hUGCTY7oK8b6tVsu4I3IzSFsEsN/tGBe/e9n3U0LY6z2vUWKrK/H2Z15vE1+C/eWXoELPUSsB5zqP9bJM8J5+Llzabt7f7hubgKvLoG1F8BMf2j6tk18SBpahz6VayLgBsfb/csQNofXVpB60FvNZ1Q1FVbjydSBBbQvCZrOJRqNh9jPjWKpFmuncte5+ljU/4atzgbGJ/J2iO6Geo/O815g5rBDejm4igzcC+D8AtqIrLg4W8+hmhHs5ui53NobQFSclLFiklgO1VAHAh9HNLPddANegu1fQbnQJ0QhcUgQHB4djGn0tgnSnefWZp8sR3Y8Ya0NBoGQQgAkIB4BsNmtIW6FQwNTUlHHlicViqFarqFQqKJfLHoLVbncTHzQaDczMdHeSU5HG+5HQMOYGgLFc0TWP2d1SqRSGhoaMSGPbTjvtNDQaDczNzZnEDXSnq9VqqFarePrpp80+R+12d+NNxi8kk0ls27bNbFZKNzluZBkMBpHJZJBKpUy/LhUMbZNKfa9B1n5WCL8ybFc0e7xsywCvIemmkLEJvR3XoK5HFGZKulm+X/tIRrVsJeqFQgHhcBiZTGaRBUfrbbsf6VzhPKGQV9gWHLtMu84qpuxYKbp8sg5+qa/90Msqp8e0/xhXpwJP22+LM71ehR+/bzabyOfzGBoa8rj62XNW55/W00/s9hKWKv40VbXWiYsutpXXzyrlB50jjPmhBYYbGNPCbAtQLcN2rbSFtIpTTbmtGRbtODLudeUE0GFGY/+rjq47WXyJc38fwK8D+BcAN+3/LgLgeXQF0AUAfg7/jVB3A1iz/30d3VilEXStUF8/QB3/Fd2EB4+iG2v0Z+huwkpRtXb/eS4ZgoPDsQ9nCepvKHG1XV78Vnxtn/teRC4Q6MYWRaNR1Ot1FItFE6fD/XoSiQQCga6rGVdLG40GyuWyIUNKOPWefK+72JP8cMWXZC4ej6NaraJQKCASiZhkDdVqFbFYDPF43GSe63S6yRD27NmDaDSK6elpFAoFpNNpVCoVEyPE1eRWq2WsTbT+aCwT4CWGfv3vJxL0XJJCOxWzlmG/79Vfeh+/6/xI9FKkd6m2LAVNDmCLF51jftYcm9jb4HWaWh1Y7C5nl7kUVIjxc6/EF37XsV4HQi9hq9/ZdeF1vUScn2imCKT1x44V0nL84o96WdmA3hYVBWP/+Ozr+Nvt1XodqH+0bRQhtALZz6GfxafXGGvZasmzBb8tAg/m+XFYQXwTwJvRdY9jUoFfhDfGZgBd17Rh69qT9v/dDX8BhP3laJa4t6KbQvtP0M1gRyR8rl0N4PMA/huAbwHIo5t57qT97+3scw4ODscuXHa4/oVNjuz/0LmCqWld7X1uAK/bi7p+pFIpBAIB5PN5Ez9Tr9cRi8UwOjqKeDyOfD5vXOhoCWo2m4jFYkYU6V4iaknodLpubBrzYych4Mp8u91N2T08POzZ1JUxPRrsnsvlkM1mMTc3h61bt2LHjh246KKLUK1WUa1WUSqVjIWJqcGZLjybzWJ8fNxYljSTnR/p0s+9SDJXlnXDRRt+RLvXeUtdD/iTWPalnmu7rNnk0a6HnyhTMcp5xQx7GkfBF62TdM9T8qptoPWRq+8Ukb3c1JZCL6HKuixFcv0EyFKwx89vnvi5V/pZyZYCLT1MP68uaOrSqFZFu51+bfJLiGG3jW2oVCooFAomE6MtwvT6XsJFy/abryyXFiAujPjF9WisIV0/1SrE30M7kYKKYF18oWDis2BbuBwOI2bR3avntwF8Ft1V1nv2/70VXdHy7/efexGAv97/vg3ghv3vCwdxv5/sv9fH0c30ZqMF4Eb5/PsAPoquxekr8n1C6kJ8EN39kRwcHByOMfS1CKKrkBIMTXZAslKtVj0ZnEgYbNKk7k1EOp1GKpXC6OioJ2taIpFAu93GwMAAUqmUyQiXyWQwNDS0iEjb7lXBYHefIW6AqgHRmlIbgBFJ69evRy6XM9aAs846yyP0KpUKotEoGo0G9uzZg927d6NUKqHdbuNHP/qRIdSRSAQ7d+40IqpUKmHPnj144YUX8Mwzz+DXf/3XjfhS955eUPGgUPLJfrOhCS2Wgh8ZVajIVesJ62/DFsC2VcRPOKhlp1qtmnixaDTqIYiRSMSIYNs1iWPsJxZsAq9xGs1m02QxfClQqxKtTbQIsn69LBbLEUJKsLUvKVSY5puumFr+wYB1tq0a7DOWa7v22ULYD2rZUeg8Z1IEjoke8xM3fhYq+5523fQ3jPF9vaxXFMntdhvFYhHpdNq04YUXXkA2m/XMV56v8Ym8notGjE2yf19dYoQjhFl0LS609HwMXRe5OQBXAngMXXFxDRaETwtdd7gP4uAzs/0LuqLmDAD/FsDr5VgH3f2NiCS6FqEL0U3esG7/9xGpC/HHcCLIweEYRaDTfa1EOf2IvhZBXLGsVCqoVqtmfxBunqquML1cq/zcPUh2SbL4mSREXUpI3plFTl1ZNO5CV7/5XTAYNFYdfq8EzhZrADwEn59JTLiyz70+kskkBgYGTLm7d+82bn5jY2OYm5tDsVg0pIiC6emnn8Y555xj3OuUyOm+TH6bamq/6nsNJtc+17Eg4dIgdPucXhYoraOddW254NjxOjtOSMksA8dtywGwsIdPL4tAILCw31QsFvNYIe0+43ta0jT2za992j/2vOf9VRiyDayPLgT4WWt4jd7rQOKIBJtiXmNaegku7Su9p85zPp+0wrHN7H87FfhSVh47xsZ2afTrb7t++rugdeWcarfbHiHjNzYKnYv8TfMTZyqs+ZuirqxDQ0NIJpOexBHcPyyZTJp4My6KqKDT+zEmabkxYw4rgCK6VhcAOA3dpARxAA8C+Ct0s7vtsa45CYeOv9v/9zJ43eLsn5o79//9nwDebx3r7K/bjTcC4XA3PfeZAH7yE+DRR19C5RwcHFYcLiaof8H/oGu12qKsXroyrySBxMSPYNkkR8mA7TZCkcPr/crlMXslWElqNBo1oomkTd2ktH5q5dB72mSbQiCVSqFarRoCmk6njZWJlqxYLGZiG/bu3YtKpYLnn38eZ5xxhgmG5qq71tsWjH79pu23XXO0XXbdewWy9yrfHkebmNrX+pXtF9PRi+D2Iu5qTbLHW9+raPWzEGif6fhqTFAvgafjYgt8v/6z0Yto233XS0zYfadzle91ry77fL972p9tAWfXj+1Qt0M7Lo/n9Lq33zzV57jXc23PLy3L/rtUn+ln9pnew+863td+flKplNkHjCJHf0v0d+tA5R8oSYrDYcRv9Pj+vsNwr+/sfx0IL6CbQvtM+a4D4C9fCfznvwDCceB/7P/+gx/siqBsFnjZy4B2G3jooZWstYODg8NBoa9FELCQrpZB0lylpJVCV4Jp3QF6x5bQbY5ZppSQ8jreVwOX6a6i6a39VvMpUEhWuP8LhZwmJNDrSVoYV8NztH1MqgAAw8PDxirEhA6nn366yXg3OzuLDRs2YGBgAO12G6eddhp+/OMf4/HHH8cTTzyBSy65xKTHrlQqnvvS3YvEyiaJWnfdcZ795ydc2Baeq9cAWLQircJUM4PZRJB1sf/q+GvZKk60PLs+9Xrdcw2FrFrqlCSzXBW79sahNrFuNBomQ5i9T4tf/ylsYaZCTmOYOp2OZz8rna9LkXWbNPsJTrsPVJCoIPbb1NSeSzqegcBCKvzp6WlPrBSJPq3CbB+tHUu5ky1Vf16r/UOh5bfAoQsvCnXhs58Fv3vb/a51sj/zGeXc5HiqBYgLG9zLjCn+1eKsQt6+v9bfwQF/jq573v+S7wIB4DsPdP8qBgeBVauA004DHnwQKJeB885bOL59O9DslcnBwcHBYeXR1yKIK6SMayBxnZ6eRiwW8+wDRKJ3oJTPuipaLBaRSCR809Ha5J/Z2VT86MaYgJds24KIcR+sI8tlG+yUtrxeY3YikYhxy0smk+h0OojFYli1apVJ6V0sFjEwMIBt27aZdNrBYBCnnHIKOp0OEokEXnzxRVNPbpiqlrR8Po9SqYT5+XkkEgnjamNbTpQcU6TaZJbtZNuUMC+VcUvFhU0G9Vpb8BzICqJjp+Sxl7UjEAiY5AKMeYlGo4vcGnX8dY7p/FDXKQ36J4FnnJEtmvxgW2xsMaQxQZrhzLby9SrXfr8UOL+5wa+OSa+/Olb2ogDLYoY2TShCy6r9fNvtZx/b9WS77Gt7WVs4hvrevg8/azKTTqdj9uPSe/vNNT9rjS6EEPzt4L38FmD0d0Ovp5hjam7OaXXt1PY5OBh8cf+LCACYLwLptFcIvetdwOrVwP/3/wGF/VkbfvrThfPOOw948smuhcjBweGIIIAVigl66UUcFfS1CFKrga5k0m+dJEdjTPwEjd/KPcmC7TLjtxqrBMgu2yaNwMJqNOtOqwCJiU1eVUTp/e1gao1XqlarRvjRYkNLWSQSQTKZRDabRSQSQbVaRbvd3Sz25JNPRjabxb59+xAMBrFmzZpF7YzH4wiFQmblmQKLMUSsj+5Boqv32ke2BcDuP6KXVUiv1fngF6ujZS618g8sbLzp57LF7+xsbRR69nyyxR6Jqt+c4th1Oh0kk0nPeRqDZffdUi5sKg4pYrPZ7KJMYdr+XgLIJvq9oIKGc5yWMmYc9HMHZFs0NkWFmfYpnxvdiFj3iNK00qyTnuc3pva59vOrbWZiE/t3RM9hn6uAUcuXjqGfFY6/XWqtUwFHgan14xjZvw32c8x72v2q42e3Z6kxd3AAAHQ6XZe3QgFIJrvfBQLd1w03AOvWdY/z+3odCAa7rnK/9EvAd7/bLcPBwcHhMKOvnbvt/7yVbOsGkHagPeB1bWFZ/KsbBfa6hw07WHu5K+S8DzNNqRXIJjNLZVezV9IZC0TCxP5IJpMYGxvDmWeeidWrVyOXy5nU3rFYDOPj4zjvvPOwd+9e7Nq1y9MfbBNdaZhQgRvIasYu7S9CA9jt8bM/91oN94vjsMUT66Ckkuf1Ej72fe301n7iV+cJr6H7X692KDn3Q7vd3RiTG9jqfXWDXZu8LtUW7QMAJgGGunT69Y1djp7jN7/9rrfjpGw3Uj8B4SdobTcszc6mCwgUN+p61+t51T7UOC3ez69dtI4EAl0LIBcQtP62tdPuW9vaa89t1o/zzLai2uPAeaiZ8TRG0q6/boxr113Hwh77AwlfBwcPstluYoRwGLjnHv9zOh0gGl0476Mf7brEfexjR7auDg4nKrhP0Eq8+hB9L4IAeP5T52qzn8DhexIH3QFd/8OPRCJmw1AlY3ZaXl7H+/oR3l5WC5uw8Hu+/MibrujTqqMrwqwLAGOtYb+wfO0PrjAPDg6ajV+bzSYymQxmZmYwPT1t9kXivSKRCObm5rBr1y5s27YN1WoV4XAY1WoVTz/9NOr1uold4mo++0czqtlpyW1Sqi+tv19/6hhp+XaZS8GP8JFI2yKaq/l+Af5qjfAjk0uB90skEkgkEqjX68bSwXG0iXqvGA2/1X2et2rVKmPJq1Qqyya32jZgITOhts3uJx0bxpFFIhFPVkO7fM45pnTW545WFYqgZDJp4t7YfxTp7De7bn6im5/ZJ+Vy2cTp2X2g40W3M1tM2IsHtpjTZ6BWq6FUKnmsXzpmXGDhy57bdn/zvfavugjy3nTf1IyA+rtiP4O9hL2Dgy86nYXXTTcBt9124POuvhrIZIBqFbj77iNbXwcHhxMOfe8OB3QtDOVyGcACsVcXEH1xBVZXnPmZZZI4kLjzOz3froffKrkS/V7nc/WaBF9JjJ+7FMslNJmAQq1GfsQUwCISVS6XsW/fPkOS9u7di+9973t45StfiWAwiNnZWTz00EN49tlnzQq4Cjb278jICNavX4/Xvva1HgFRqVR8V6K1bvxOrSX2uNl9qe2nK5VfHJYf1O1RUyqz//2IH0Ws9p9anXTu8fxexNEm5OxTzrtCoWDIPi1vWh+N1bLnslp57L7UcVOXLZZp19cWSiTmdEVTUW8LebafLl3282L3RTAYRLlc9jwjrD+zMuoeNhpbY2dQZD/63cPPvZRCyk6MQpcz7W8Vour6WSqVTF9oCmqd2ypyelkGtd7qFqt97Dev7N8CtVb2WmjheAILST8Yk7bUb4aDw7JQqwH741XxC78AvPDC0udns10LUS4HvOENh7t2Dg4nLjr7XytRTh+ir0UQoQRPPwMLgqEXmVXy50eW9FyFfa7tcmKvoiq51hgBJVR+99MVWd5LCb8fWeVf3TVe4xB0pZrnkahx1T2TyWB2dhbPPfcczj33XFSrVezevRtPPfUUtm/fbgiTkkPeb2ZmBs1mE/Pz8xgcHPQQNg3A9rNQkcTaAfS9xkDb69f3dhl+fazWlaVSUGvZfm55vcih7XJmCx/7Hjr2uiGwX1+oANW57Gf10/JVwPoJPBUwWtdeY+DXx36xMH4xOX7XqcXHvp/e1z7PFldaf7vuFBYUgVo+32sbtEw95icUODb8bD+3fuNgL7RwHPySYNjiTYWl3VZ7Icb+rfFbCNJ6H+i30MFhWfjc57p/P/hBYP365V1z+eXAl74EvPGNh69eDg4nMk5wEdTX7nCAf8ICAJ6VY3sF228Fn9f6raYvtXJtxzzotXqOJm5QAm1bFWyxpsSb11Io2FBiqNnNeK5apjqdjnGHCQQCntiKSCSCk046Cel0Gjt37sSOHTvw05/+FD/5yU+wd+9e0792O0ls5+fnsWPHDmzduhXFYtEIK1pp6vW6R4TpuLXb3k1VbaGznO9s4q5jz/6xiaCfS5nCj0CrhUXjT+x6qHumbeHSuUUXL7tMut7ZLpd6zM89yo+806WR52lGPn6nG6eqeNM2qYCyx5DvbbdALYdZDFUoqOuXX9v4nGjcj9aHblya+czuA62jvTEoX7brGRcR1AVS66B93253s/kxWyTPZbrzXvPUT3Dw/ioy9Rm368u6atpu2yWO7bbjFzkv7PFT1zudww4OB41t24C77urG/SwXmQxw5ZULAsrBwcFhBdHXliAlmerewf/gg8Eg4vE4ZmdnPYKD/+nTxYpkyHYh87NWEDaBta06rA+JB4WIvWLPuvi5t/gRSF31JkFUkNS0Wi08//zzJoU1yyJJqlQqRvjwvkqQTz31VKxevRoTExP49re/bfotnU5jYGDAxE2EQiHkcjkjuMrlMhqNBmZmZnDHHXfguuuuw9lnn43169cb8kUBlk6nPdYYZrJjAgW2zR6HA61Ek6iq9U1JIfeQYcIIwBszYQsNYLHbHPtJ5xqvVXEZDoc9+wGVy2XPvXgN547GynCMtP1KWgEYwahzmG5nStRrtZpnjnDu0/VJhYPOa7X0qbVCk0Cwj+mGqHNZrRKsk7aL7db+87Pe2uKFYj0SiZj2M55IE4JofTQhAc9hiu14PG7ayv5WQaqWolqtZp4VpsVXoabzmq6CPD8QCJh9u+x5zKx5al2iCOm16KGLNWxTMpk0GQZ5jG1geRxvClkKqU6ng3379qHZbGJgYMAspoRCIU+yFQeHQ8LzzwP33de1Bi0XySTwpjd1Xeg+/vFu7JCDg8OKINBZoRTZffpY9rUIAuAhOOqmYq+EKmGwXVH4nZ/bh8bckOzY5+l9VFAoedNVa79YBLsOahlR4q+uVSqobAtEOBxGJpNZlHbbtgAwrqPV6m7UOjg4iMHBQUNSJyYmkEgkPBYdFSuxWAzxeBztdttsyBoOh1Eul431KBQKYXh42BAqv6QPtsVB+0CP25/tAG57TMLh8KJMWZFIxEP6l3Ijs+cSx48EVUWcWoLsMbbr53cvEmaOXygUQqlUMm5xdsyYkltujMssgDoX7HkCLCwc6OIA+8kW7NpujQNSYq710QUJ/aybeHJzYd7b3ijXtpxqnA/30yKJZ6p2FSoU+oylYp1VbHHTUMa7UUSGQiFPOzmPmAWxVCqZjIq1Wg2NRsNjCSyVSkZox+Nxj9CiWLTd+NrtttkYl+KO9WGbdXHFtuBRBKuYpuDRrIwau6Vzj9aqdDptFleYlIL95yfEHBxWFPffD0xNdfcMetnLFr6PRICPfKQrov7mbxbiixwcHF4aTnB3uL4WQSQAXLklAVDYIkHFELCwg7uSahIqm4RRLCg58bPscOVXSYqu/CtZVlJrCyDWn0RJrSP2NXod28F9eWxBBmBRil5aN9LpNFKplCF1uVzOZOAiEeN5mUzG7DdTr9dRKpUwNjZmNmXNZDLYvXs3nnvuOYyPj2NwcBDpdNqzl47WXffmoSBgu20XHFtIkQTSMsC6JhIJj5DR69S6Y/e9kkMl1zy/VCoBgBEdJMok9RxfzgXb3Ynz0I5H0XicTseb6pyioddc1rbY1hWFPfdsywPnnbpR2pZOEm977qnYtuOrWA5FBo83Gg1PAhLOBd5PMz7aYol15jksk+mrOafs+Bqew/ZTFKoI0t+UWCyGarWKcrls9gZiH+niQCgUMpsQc06q2FJ3Ph1LbSPgdVPUfvBbZGH7bUFru/LyOv1t0znIxQ3Os0QiYSyZOmYHssQ6OBw0fvhDoNUC3vMe4JFHgDe/GfjN3+xupHr++d1zAgHg//wfYHIS+N73uhnkHBwcHF4C+loEkZDpyjeJi5J8Tf9MQkJSQnJPywmARSvLJDUDAwPGzYQuL3R9SiQSRiQpuSJZsjN1qfuM7iSvq/0kKWodsAUX/6q7lFqc+LLT76qrjwo7Eh66C8bjcaRSKSOqdu3ahdHRUZx99tk466yzkMlkjJtTqVTC17/+dVQqFTSbTaxbtw7PPfccHnvsMTz99NO4+uqrsXr1asRiMdNnSuDi8ThqtRqKxSLS6XTP7HpM48u+BBYsG4lEAkA3LqlYLGJ4eNiTnUute9Vq1ZzPvuHYEyr8SHZrtRqef/55rF69GtlsFtFo1FhhwuGwES0k6lxRp1CKxWKeY+piRLENdPfzmZqaMmmzV69ebea1kmW6WnFO21ZBWyypex4JuYoLum2xDFrO2Afq3qZihPOf96U1S93l1IWM1zebTUSjUQ9553V8bnXDX21jvV435emig1r9qtUqarUaYrGY6S/OG6bBZkp5FYc24R8YGDDzLxAIGKvQ0NDQooxvaj1TIabWUPYThbbG5KhQ63S66cUpOtkPfgsJHMN6vW7mWqfTQaVS8SyocGz0vqxzKpVCp9NBLpcDsLAQkEgkPKn3HRwOCfU6sGsXMD4O7NzZ/e7yy4H9C0sAuumx7767aw366lcXvl+zpmstevWrgR/8oJt1zsHB4dDhLEH9CxIzJRS93JBIGHTvH14Ti8UAdFNEKyGNRCLGRSWbzaJarRoCSdce1oMuMfysoocEySaoXK0eGhryEDclrerKZ4sf1t9uo16nK9wacK7ue5pFjuc3m00UCgVMTk5i1apVZnX41FNPxctf/nJD/rW+uVwOl112Gfbu3YvJyUk8+uijZr+VcrmMr33ta7jxxhuxZs0adDodDA4OelbB6WJFQaGxHSRqamEgbAsYAGSzWUOuSXrZbo6DEnglkPyrFhoV2/V6HZOTkxgdHTXEWF201P2OfWO7K+mcoLhtNpuGgHY6HcTjcezbtw/JZBIjIyMeYq3kWsdehS8Fkx1jw/mhQowigvPBdgNkGZzXLF+TNWj/87O+Z7n1eh3xeNxzvs5z+xlWtzoVqKwXY4TUoqPWPsZ+sZ26GKEihOOnddY62osOXEzg2LI8dRuzBbxaBflM6qKNDe0Hrbtad+x+AmCeTcbnzc/PIxqNYu3atabvKGi5+MJnhHPY/o2w55CDwyFhyxbg0ku7f9et68b87N/iYhF++tPuOUSzCYRCwAMPAK99bVcQOTdNB4dDhosJ6mMo0VFiAPT+z1pFg5JD/qU7FbCwimsHsTO4m8cpxli+HS/B65RUA/BYBRTq+2/HC2mblPgoEdV+0L/qjqRxARqHQTLJFfvBwUEEg0Hk83m0Wi1s2LABg4ODi/Zf4T3T6bTHNWxwcBA7duzAzp07UalU8Mwzz6DdbmPVqlUmfXYwGEQikfAQPRJWxjbQOqFt8mu3tkc3fuRx1ovXqNhRYmnPJe1Xv/gV3seO2+F80Wu4Sq+JBfziPDgOjMmwCbltXeSc1X7wq7+OPfvAfhZ0/rFPKFS1bb0Isd98tftT26ljYLs69rI8KDnXNmk79Fm2j9vjbosKbavGJWlCAdtVkvfS+6v1jHOadVcB4vdM2b89/F7j0fTethjiIg/jA+3+08UALVvFrT5P6h7q4HDI2L4dGB3turjl88DgIFAoLH2NzGsAwNe+Btx8M/Dnf3746ung4HBc47gRQeraZRM9wLvZoJ+FiGVRlKhbnRJOZqAqFovGgqTk1yZTWlcSCBJeknSNA9F7s968h02uKTZYvrbFT+gwNTVdi3iNknn2EV2gEokEstksnnnmGVQqFYyNjfkSJ9aLQiiVSmHt2rU4+eST8ZOf/ATz8/Oo1Wp4/PHHMTMzg3PPPRdDQ0PG/S4Wi5l+0NVooOuylMlkFrkKsh+03myrbVnjdRS5tpXHjrWyY3iABZc7zodGo2FcvNSioeOnc4/9Zq/+q5DSe3CsKIAotHh/zdzlZyGwxY5aKuyNQHU8OTfthBpMnqHiTvtHRYXOU763v7OfUa2rlmG7qalQVfdAZlez62RbAe3783nR58+eZxxrCvJoNOpxc9U26meNy2G/cn7RfZAiWUWKnxDXuCaNf+olNvk+lUohlUotGiPOWYo0XqfzgfelpcyJIIcVRafTTXowO9tNh30g2ELos58FTjsNeO97D0/9HByOd3QC3ddKlNOH6GsRRKIQjUYxNzeH2dlZ7Nu3zyOCQqEQKpWKIRmJRMKsxEejUVQqFSMKGHvAdLG6Qq8ruJFIBAMDA0in04bQkCSpi5GSE5ImChcl6bbgUVFH2NYPFUL2Kj+JY7vdDeCvVqsIBoMoFArodLrWrrVr15prNN6F9S2VSkacDAwM4Nxzz0WhUMCuXbswPT2NoaEhDA0NeVbIdcUc6MYWZLNZrFmzBmeeeSZ+8IMf4MUXXzTJErZu3YqNGzfi7LPP9qyWAzBEneNmt1/vqZYEAJ4U0XaKZx0nujbapFk/q7hWEa2xQyoU/MSUknZNVsGYNHUFI9lkm5nZi3OSbmT2c+B3T7WAaVs0BkQXB7TNKsR0XDUWxbYs2NYItUrwev7lAgLgtcD5QV1Uda7zs9bHPq7X2JYftbTSKkNo/AytICxDY+w0pTXL1vvyL2PJGJ/I3xs7zbhey2eYCy9aJ/6GaVY6jpXGldl9ZNdRx45iXGO7tF/1PAeHFUOn04310Xn5u7+7vDTawSDw3/4bcNJJwH/5L4evjg4OxytcTNDxgVQqhXQ6jXQ6DQAoFAqG8NKKQLLAQOpAIODZQ0cDv5nlScULYw94LTc/ZDpepsKNxWIoFApmlZhiSwm27ZqmZIzWEQo1kiQNfFcogdHVbQAmuQE3cFQXIiW5JHKa6YpEjcQ1EolgzZo1CAQC2LdvH5599lnkcjlks1ljsSApY1npdBoTExOo1+uYn59Hu91GsVjE5OQknn76aczNzeHxxx/HL//yL3uEoya7YB/5EVu10qllQwUex0njHtTywr7QfuN4cMzs+09MTCAej5tr9J62JUlJo505zibsNqGdn59HIpEw7oFq2fBzy9P+4VywRbffSr5ahvwsC2pNUcFt119hWzFYDxWWtkXHtqpwDHRcVOD5WexswaxWLS3Lvl6fLbXsaX3suD9ahtSCZM9Tzg0KJrXmqZBk3ZrNphFMnLNMyc1+4+INLbYKtZLyPvl8HgCQy+WM8NLYNd6fFibbaq79YotmB4eXjGLR+/mTnwT+6q8Wn/fUU92YIEUsBrz+9cBXvtL96+Dg4LBMHBciiERHCYGuWJMwkuQp2WF6Z8C7sSGhhNFvVV3dZpRc8a9aIFgWxQKP225LSkJ0tZoE376H1oPH1fWH7VbxU6lUPH1Yr9dNpqxqterJdqU717M9XNluNBool8uo1+tmFZmr5LS2xGIxrFq1Cqeddhr27dtnRFY+n0elUkG1WsX09DRWrVrlce1iXf2IsRJNdROyrWP2Kjavta02elwJpE0SiVQq5YlTUpc3vzLs9tgWJ7t9/MwVf5swc26ogLaFnFrB/O5jW1TsuttzjOWoZaWXALKhz52+/KxHftf61c3uRz+Lh/3ZL/6Lc8Gee2qB0vG1RVYvcW5/x+fPvrd9nt1W26WPfakC2K/PVAQV95PMdDptRJC9YKBi0G6fLSAdHA4r5ua6LxuXX76QKnt/kg8A3VTav/RLwN//PXD99Ueqlg4OfQ+XGKGP4fefP/+T1nSulUrFswJrCxuKEAoO/sfP1WOSbAorruoCCwSce+n4WRyArsjgSu7o6KhHZDDtti2AaJ1S645tUVJSwsxqwML+NTah5Tl2sHmpVPIEbJPw0RUL6BLBarWKZDKJWCyGiYkJpFIpzM3NIZ/Po1arGcEXCAQwNDRkUocPDg7iggsuwEMPPQQAGBkZwczMDCqVCmZmZvD0008jl8uZ2AUSL5tgsh726rUdX0M3IsZd8Xyu3utxBcu2yaXOE46LbriqhNUPSxFVv2x3tvhS65xN+pWoah9x3GyXqeUIH62z1kXFpn5vk2Otk71QwPmtCQb8+sfuD8Le38fPOsr72eOoIkhFhVqj9L52jJD+LqhFrFeMnJ+4Yt1CoRCq+/c6CQQWLL1021V3OKaUZxvS6bRZtPATV9qGVquF8v7sW41Gw7j/8jN/ywi/sQSwqJ0ODkccDz7Y/fuudwG5HPDrv95Nlw10Y4quvhr4i7/ofr75ZsDFrzk4LA3nDte/4H/WdEnhe7qSkQgyeD0cDhvXKJJjmwTV63UPmbAJpKZt1jqQiNMNrFQqIZfLmUBypnymRchOAqD7dugKNF36AJi9YNTlzCat/E4TC9gr9xocTuRyORSLRZOEQN3+7JTPjOdot9soFAo4+eST0el0UCqVkM1mTftssZnNZjE2NoZarYb5+Xl0Oh3Mzs5ienoaX/7yl9Fut3HqqadiYmLCE9/Be7F9tVrNuPYFAgGzh4q68vBcOzOXijx1cbMtGzZpVZCoakplFWHM+MUyKCQ5n2wLiApn7TMS8Fgshmw266lDu902gtdeqac4oKi1rZt6b7U4shxN4KGWStuNza6Plm1b36LRqCH8KoZUqNgijNC0435WNAohPxHDtqs4tu+lcVtsn+3OqLFT+jywf+3v9NnR35NgMGgESTab9YzBzMyMcYFjn7FukUgE5XLZ83tFMWM/1+pyyTl36qmnehZy9JjdF5xXqVTK0zelUslZgxyODfz933f/zsx0N1m9/PLu52QSuOmmbpzR888D//t/u01VHRwceqKvRRCweMWZJECP2ZsealpdXQFWcggs3oNHib1fem2SunA4bDYYJZHld7alwV411jKBxW5W2hZbzOgqu77v9VnvTVGoCQn0nn4xD8Fg0JA2ur7p/dknXMnudDoYHh5Gs9lEMplEqVRCMplEMBhEtVrFli1bEAgEsHbt2kXuR0pM1bJjWxpUyOoKPMfPzqTWy3KjZbMdvawmtiDQDVNtUaX3tFfs/axfahm0oeXYbl6sRy83wV5lsR+13mpZ69U/2gY95neu3/zX82zoGHJvLY3VUvGizyxdNHmdbRnVMdL+0fqxbE2Vr/1o15/Pil13WwgBCxYt1kHjvhiPSKj1i79Z2pf2/lT2fNAFIzsOStvOc9hOino7ds7B4ZjAl7/cFUHxOPCLv7jwfSAAfPzj3c1Y//7vD5x+28HhRMUKucM5S9BRgE207O/0P3p+BhZWlnUzTQC+hFhXge1Vej+QNMbjcY9VhwTdJs761ybHGgivpNSun95bv1dCqCSMpMm+Fy0besyPxCl5tbPgKVHW4H2Wl8lk0Ol0kEqlUCgUkEwmzaasL7zwAoaHh3HmmWciEOhuvppIJBat8NuiVutJdzyKOpsA+rmeaf/Zc0dFr7ZfrRH2XybVsC1LKlq1H/3EkD2mfoJZx0QTemjwuiYQsK2XWmfbTcyvf9Ti4Hfcrq9C57Z9T1vkLWVp4HNru/fZZdGiwf2n7GdL+19jq+x5rufo74hfH6rgZDvsz8xUabviATDucGyjpsFWwWbXme21RabWUcdX564tgvR3jvdUy7WDwzGHr361m1jhT/8UOOcc77E77+we+/rXu/sROTg4eOHc4foXSkq4oaQSeI2RAeBZQa1Wq6hUKhgZGTHn2+JAA+6DwaBxu1LCZbvUUPjEYjFUq1VDyJlBSgmVEmB7pbWX5cEmYHaAtJImttmGTVDVhZBuZkq6SNrUMqNklsKjVCoBgBEgrAsFIDeHHR4exhlnnIELLrgA27dvxzPPPIMdO3YgFArhpz/9KX7+858jkUjgyiuvxBlnnIGBgQEztiT83M/JzlCm/ajk009wKJYi3pwLakVisggVwyoEmVUP6O5zRHfMQCBgEiqoOLMFDrDgDqf3YLk6Zp1OB4VCwcSSpFIpBAIB1Ot1YxVTkWuTfWBhrxhmMGNd7Hmgfap926tfbZHpNwZ+QlTroBaOYDBo4t0CgYAnpkVdH22BoVYeFTa8h2anY1/ZiSVsscvU1a1Wy8Qf+i2U8Deo1WrhmWeewdq1azEwMOAZb+2bUCiEZDJp3GDZTtaF1k79HdLFB1vA2X1vC1G687VaLaRSKdPn7E/ui8Q6LGU9dXA4Kvj2t7spsn/848XHvvhF4MYbu1aj/a6oDg4ODkCfiyCFigqbVOlGiEz1nEgkDJHmeSR9JAyMM6Jomp6eRiaT8cTEKJRAkbDxvcarKKli3Iq6oPVyl+JqvhIZXfm2V3vZPj+Cqi4vJOHVahXlchmpVMoQQHW70X6yXdOCwW5yCB6nQLDd/zTGodPp4KSTTkIkEsHDDz9shGmhUMCqVavwta99DQ899BDOPvtsXHnllcbqZO8xo6vXKtSUAPMY+9JeMffrd9sapFY5FYUqVnitJtFg7Abnge165hc4z3apmOTc1LYAXcvT4OCg+Y5kldYo23VKs9rpHABgxoVgnBvrR/JtW4x6iUgl2xrrZm8iquf3Eu5+CwRM/c6+ZX/Ylk7bQqUCD4CZU7ZIsp9BBdtkP5NsH+vC+kYiEZx99tmLkp+wfnYZ+h37S3+TFGqpsZ95231PYwU1qQKzEKrgZXwbFzq0Lx0cjim02934n/1bF3jw+c8D73wncPvtwP7fCQcHB5zwlqC+9m9QQmP/x6wpnZVc8D92Hrf3yvArH1ggCbZLiIoIElwSDSXLWo6SZnslVkmPbgiqbfUj3RqITWuVWq1I2DSWSYmoWrzsvYb0nrqSr3UFFtx59H5K+jUZhArNbDaLs846C6lUypCvcrmM6elp7Ny5E48//jiee+455PN5T7rzXvDrn17jqm3TNtn9bvcHr2N/60a5dgY623JnCzQtz74v5y/jMjhetiVDy+T1oVBoUTINzhG7f5Qga5/Y887u96XqruXqnGB/2fPItlBof9p9ou3md7a4sONi7Lb6tUOv089MB28vHthJQ4CFlPJ8KWxrda97djrdhZd6vW7GXe+v0L7XMvzGoNfvi84nXajRa/S39FjDgw8+iNe97nWYmJhAIBDAl7/8Zc/xt7zlLR4hGAgEcPXVV3vOmZmZwY033ohsNouBgQHcdNNNJrW4Qx/gpz8FNmzoJkXww6c+BfzhHx7RKjk4HOtgiuyVePUj+l4E9frPX338lbAwVqBWq6FWq3n2wAEWu7iRnIRCIWSz2SVX0TWAWFelNdMaj9lWEj9hRGLt58/vR3SCwaAnA12lUjH1oOuO7uGjwiYYDBq3F3Ux0/rYbjsqBLhKrRY57XN19bFJYDwex0UXXYSRkREkk0mEw2GUy2U0Gg0Ui0U8/fTTeOSRR7B3717fRAG2kNA624LH7jM/EmqLqF6iSQk9CSrvq8dssapCyBbCdjt4Dl3blMjbrpu2lSkajSIejxtXUc5D+56cp3rMfmb48nPTtM9RIaPkmn3G586vf7WPVUj4PaM2Mbfvpe6sWh8VkToOdtIAbRP3s6KLmJ0uWt+rqyTTWOucsa/xe/7b7TbK5bIRX/r7YveXLfa0H/m9/Z3+vtjiUBcatEydH8caSqUSXvayl+H222/vec7VV1+N3bt3m9df//Vfe47feOONePzxx3H//ffj3nvvxYMPPoibb775cFfdYSUxOQns32bBF7/920CjAfzkJ0euTg4ODscs+todjq4oGtDP/7A1oxkAz1/68ZNEMUMZsHjVVAmEEnBbHOiqNe9lk3Ul20rqNIifJIsEje5p0WgU0WgU5XK5p3Wj3W6bmJ5Go2FSErdaLUM8NYsUSSHdiUKhkMd1y2/1V2MnOAZ2enLWhy407FNeq4IsHA4jGo3itNNOQ6VSweOPP46f/OQnqFQqnsxeP/nJTzA0NITx8XET88L76PjSksV5ocf0fNv1jG1eyoqk9yK51pTXJPj1eh2xWMxzjHNB01r3ctfT8bQtRpwb9tzmOByoXN5b22k/O7YFSPurlxXAT/jYlhutq+2aaL/nPf2+4187ro7PH/vA3v/GbpvftayvLZjsxBg6n+0+YFr8er2OyclJJJNJpFIpkx1SXdc0Yx2/5/jkcjlP2Xxe/MbGFnW2BdLuX8Z+UfS0220TX0WxrecACy5+x6I73DXXXINrrrlmyXNisRjGx8d9jz355JO477778KMf/Qgvf/nLAQB/8id/gte+9rX4oz/6I0xMTKx4nR0OEyoVYHCw+/655xbeA0Aw2H2df343vXajAYyNHZ16Ojg4HHX0tQiyVzi5Gq5EnIRVzyGJUeJA2Kuc9go0r1ULka4M2/WyV1NZD7WWqGWnVCqZDUYZnK17Fyn51VgYkiQlgboHUbu9kMKa5fA9y2H/MbOdtoF9qO44LEdjVWyLiva9JqpgfZTsn3zyyaZOjz/+OEqlkiF+hUIBP/nJT1AsFnH55ZebzSRtcq4ZunTzVyXIao3RdqqroLZP68j32t/1et0QbgpaHWsKM66+26JDLQ8cC35meZpZkOPE72iF65V9kHEdSuB1TAOBgK+Fgdfbljuth85d+zwKLO1P9pf2pZ/A0gUDPYflELVazbOJKGOadIzsWLte92O/1ut1szDA8zOZjMeNlt/b8Uxqael0OiapBxcK9LfC7i87yYfC/o2yF2tUxAFYNOe1v3WBKBQKeZJnqCXMtlipgO9HfOc738Ho6CgGBwdx+eWX4+Mf/ziGh4cBAJs3b8bAwIARQABwxRVXIBgM4gc/+AHe+MY3+pZJjwIi7zKQHRuYm+v+vfhi4F/+BTj1VO/xUKgrjjod4Oc/B847z8UKOZyYcDFB/Q8lM/ZKvk189LMdQ6Av2zXFdvlhGTZ50LJ61dXPwkDypW4qSuA1vsavjepuQwKmyRbULY91V2uJTXrsvtUkD0oy7WvVTUjFhpalwlAJWiaTwfj4OE499VQMDAwY0UYxuGvXLjz55JPYvn27CfrX1WnbkmFbGXq5ANn9pvNF22eXS5c3dUPTZAZ0h1KCbbtCavnqRqVjy/MYuE7ipdYhvzgTfSZsVy9bCKlrnNZD59dSz4ndFvtaFf5Ni2zY9dXr/GJgdD75zVW9p1+f23W17+cXi6fuhLYIUFdIe27F43FjobMFn95bY4207zjHVKD79Zc9P3sds/vYjl3SuaF9utRvWj/g6quvxl/91V/hm9/8Jv7gD/4ADzzwAK655hrT7snJSYyOjnquCYfDGBoawuTkZM9yb7vtNuRyOfNau3btYW2Hw0Fi61bgTW8CfvQj/+OBAHDmmcA//zMgllcHB4cTA31vCeJ/1nbgt03iNMsWV625OgssEH+b6Nv3UXHCYyS59h40fhYUnq9WqEajYWIHIpEI6vW6SSedSqWMOKKbCttlr/LaUEuBWpEoUHTTWK2v7a6kYkbJklpHAK8bEUWXxsmEQiEkEgkEAgHTTpbNemWzWZxyyil45plnjIDg9cViEbVaDQ8//DCGhoYQiUTQai2kJ+YYcSWfFjBFtVo1QoV9ZFtzCJvw2ySSFhZNCMH60hVwdnbWWCdIaHmeWvf8LEWsbyAQQCKRQDweR7lcNgHzFIm2pZP35/j3Snahc4fHo9EoqtWqsT6pxcB2L1RLpG1Zs4WKX6yd7aZHsC5qMbGtthQI9njxWfETR6wXx5Kuofo8ATCi27a+MtU9rWqsQ3l/2l1NQKH9o78nui8Z20iRFolEPKntO51ughCm3LfbY89LdUG0xSvLVKsf57D+PqiA72X16kcxdMMNN5j3559/Pi644AKceuqp+M53voPXvOY1h1zu+973Ptx6663mcz6fd0LoWMPmzcCHPgS8+93AL/2S/zlXXAH88R93s8t95jPAo48e2To6OBwlrFRSgxMmMcKxloVHSQ13k+eO8sDCf9jqihWLxcxeI+pCpOSE7zWjmr1CqivHGmdDscJgdnsVVoURXVEYxJ5MJk1MQaVSMSKE904mkwBgXDDU9U37nMdI1Hi+usDpXxVMKiqUOOmeSySNlUrFkC3GIHEsarWaJ8PVvn37zNjo+KjbU6fTQSKRwBvf+Eb85m/+Jm6++WZcffXVxkWw0+ngkUcewZ/92Z/h//7f/4stW7Zg165dRjCVSqVFq/vMNPfCCy/g0UcfxfT0NNrtbgzEyMgI4vG4aQctB5rBzI4vYttisZhJC65udhQdkUgEIyMjRrxx7vF4pVIxyStsIVmtVpHP540oisfjqNVqiMfj5rlh4LySWrVE2f3LMWs0GmZucu7qvLatHRxHCjjO20qlYuaNtkHnjL0YEQwGTaxJL+uELkiwLZomnGMViUQQi8VMHJYKFo4vXdH0mWAf1et1VKtVMzaMu/PLPsi+43NA4aIxiRRJfqLZFlRse6lUMs+Viiv2AecjF0v0d8a2GnFcK5WKxzqlyTtsy2UoFPLssabxemx3o9FApVJZlAyin3HKKadgZGQEzz77LABgfHwcU1NTnnOazSZmZmZ6xhEB3ec5m816Xg7HIP75n4HHH1/6nDe/GfjN3wT+238DfuEXjky9HByOBXRW4NWnOGhLELPw/MZv/Aauu+4633Ouvvpq3HnnneazrmIC3Sw8u3fvxv33349Go4G3vvWtuPnmm3HPPfccVF2UqChZojWChFWD+e2NPwF/lw+ujip43La8UMSQRACLN2e0od9rVjlaSZRQk6wVCgWMj48bcs5geF0ppzCz3blI2pWMav1JdjRphN03tvsa+1pJIwl+p9PxrGxzjGzXMXssVZSR5IbDYezYsQN79+7F/Pw85ubmMD09bVzD9u7di3POOcckTSAxJwGk8EqlUhgdHTUxT+12G6VSCe12N6GE1tV2a7LJup1tzT5XBSYD4nX+qHDiXOGGujwnFot5NuHU62nJ1OQL9hzV+uo5+p5jRaKtFhE7oYFulmnfj+Jf2+VnLWT5tsVCz9Hx03Ps+Wg/j3ZcniYxseew1sHup15QIQXALLpQnGrf+D33gUAAlUrFWKApSlhPtTKxLvF43CRM0HnlZwXWZ0j7yS/pi7bZtnirsPKb10v1Ub9gx44dmJ6exurVqwEAGzduxNzcHLZs2YKLLroIAPCtb30L7XYbl1xyydGsqsNK4bHHuq/zz1/6vN/4jW6s0Kc/7SxCDg7HOQ5aBB1rWXj4H7KuGvM/bvW157m9MjrpeyWi+p8+ibNNMAEvcbNJgp2Mgd+xPLrCkQirCCL5qNVqmJ2dRSaT8bgeabC/HxnmexUfSlxZJ64ca3yAXx8RtrucTciUCCqp1/5RwmULOX7HlNmnn366EQn5fB71eh2zs7OoVqvYt28fwuEwGo0GxvZn+tEAeQq1RCKBdDrt2URWRVClUvEIY1NEQGUAAHn1SURBVNtKpaLWTiPsJ57ZFlordJ8fe76pBYnXclNf3exUr6fw9pu7NmxXUHt8bFFCIq3zyo8YK+lXFzIVyFo+x0MzoWm9VQyy/F5CU+cJnymdW7ogon2u97fL43U69/3IP8+nZcjuH7/5wD7ifavVqkfE6X0476LRqCnT7znTsSds8af9bC/MqNVJxas9//X6Y1EEFYtFY9UBgOeffx6PPPIIhoaGMDQ0hI985CO4/vrrMT4+jq1bt+I973kPTjvtNFx11VUAgLPPPhtXX3013va2t+Gzn/0sGo0GbrnlFtxwww0uM9zxgj/7M6BY7LrFxePAWWf1Pvemm4BMBnjf+7oZ5hwcjlec4IkRDktM0Epn4TlQBp6lsm0pgSCxnJ+fB7CQyUvd00iQ6B+vJIYrt4B3h3nAm/WJZav7DM/pdDqL3Maq1SpSqZRxhePqMMsIBoNIp9OeGBC64dCKpATJFm20cBQKBUQiEbNqrQHjjENKJBImI5uumqt4o0gjwSV5ZD8ylqRcLntWyO24Gdsq5Ee2KVzPOussBAIBk/1vz549ZvV/fn4eDz74IL773e8iGAziV37lV7Bu3TojoJLJpDlXLRQcH9adq+66El6tVs05zBhGlyEAxtpFAcN5o/OwWCya961Wy4wrXdOY1rxarRoLD2OKuHeTpjbni0KP7VhqE1kdI5voKtRywDHjM6Kigv2j+05x7Eni6bJmWwBDoRCKxaLH/YzPhz5H/MtYKFo+OVftOtvijuI+Go2aOnAOV6tVk1WOz0I+nzdxeLye8zSRSHgEC38DIpEIyuXyIgHP8zRbJQAkk0lTJq2Eaj0jbOsfszZy7gDw7BvFecg+10WAer1u5pRt5dGFIdaLfRkOh5HP5z0xdMeqCPrXf/1X/JLEezBO581vfjPuuOMOPProo7j77rsxNzeHiYkJXHnllfjYxz7m8VL4/Oc/j1tuuQWvec1rEAwGcf311+PTn/70EW+Lw2HEPfd0X2efDXz7293ECFZCDIP/9J+AkRHgV36lm07bweE4xIkeE7TiIujqq6/Gddddhw0bNmDr1q14//vfj2uuuQabN29GKBQ6pCw8t912Gz7ykY8s+l7/Q1cxQxJOC0oqlTLxCdPT02bvnWw2u8j9gySP/+krOVciGQgEjOsYABNUT0FWq9UWiTJeF4vFkEgk0Gq1UCqVTNIAXUXnuaFQyKwYp1IpNBoNFAoFNBoNrFq1ykO41Z2L98vn85ibm8POnTtx+umnY3Bw0CNeSP6VECo5VmHF+vht4qor2Pw+nU4b64euuiux1jiHpQhWMpnE6tWrPbE0KkS4uWooFMLc3ByGhoaQTqeN65tN+DXRA+/JpAm6ik9CqMHuHN9isWisFnodCbyKFc34p/OCQfgam6Er9/v27UOr1fIIYJJVJjBgOzj/NL5LhbCKK51jnU7H1I/PD6HxRrYVwbby2CLMXojQ72nhYntUxOuc4qJBqVQCAKRSKU+K52Aw6LFAadvtxQF1gdP+ZBsSiYQRs3pdLBYzZfGlbaFlUff4ooBjGni2ifM1EAigWCx63NqY6IPzTBdvgsEgdu3ahXQ6bc7ziyEKBoPG7ZX34zPL95yjapXVRQKNH2RZvOZYFUGXXXbZIkGv+Jd/+ZcDljE0NHTQLtkOfYonnwTGx4FwuGsdika7gsjG5ZcDDz0EXHBBd08hBweH4worLoIORxaepTLw2K459XrdE/+i/zGSaNjBz0rUbGIAePeeUcuS/Z+ubYVRsaGxSGp90Fgmv4xz9t4/uuKthI/XkXwpMaY1RJNAaH1tK4O2ndDVeSWONnnVGBW/uAu1KNlZvAibZLLcTCaDsbExTE9Po9lsolgsGoGoloMdO3ag0+kgn88jk8lgaGjIbAJrC1PbImJbEZXAa0wMEwvYrkydTscEzhMqFuyMW7ZblgpCjjszkGnmLs4LipRKpWKsSEpc7fbYJJ/3LhQKaLfbSKfTqNfrZhx5LsWTjrH93KgVjNZMCnzWp1wuo1AoGCuKJhVQMq91L5fLRvDOzc0Z10fG+2liAc3EVygUEI/HEY/HjUuiLjaoANbnXsUUnzcdNzvGZt++fSZbn/426EKD3od1pHVL+1fnm4oOLmhwvGnZ0gUbutWWy2XzPPi5A1N067PLBAyxWAzFYnGRhTgUCpkkHg4Oxw2aza5rXK3WFUJ+OOssYHIS2O/N4uBwXMG5wx1eaBae17zmNYeUhScWiy1KrkAoqSdZnJubw+jo6CLCHw6HEY/HPWlqlTwrmVVCSnJJEsvyeK4GLdur5SRkGkhPwqFl8zivYb1oUSJJDAaDJkOcrhyzrSTHdJuJRqPGKsINRtUVRi0i9oq17dZTr9dRLpeRzWYNiVKCByzEMNDlSDNzqeBgBiqSTL2fWue0r1OpFCKRCCqViumXdrttsoWx35555hlMTU2ZbE3/5t/8G6xatcq4xal1xLYA8j0FKIkpU0ezThQ6dlp0Wi50o1ada6w3BQBJJokoXRF5D5LWubk5pNNpz15BahViPBMtjBQy6oKo81KteKFQCNPT0+h0OohGoyZ4PxKJoFqtetwneQ+1ULLPKpWKcfcqlUpIp9OLYr2Y2IIWFt4nHo+bTIIqPoLBoHGdA4D5+Xmk02mTEY6uchQG7LtqtYrdu3djYGDAuJnyuaDwKhaLHmuYpszXPYE4xrT88hni/bZt24bh4WHEYjFTf877ubk54+7I3w9eF4lEUCgUjAikW53OTWDBIkPRX61WkclkUCqVEA6HkcvlUKvVkM/njXWQQpl9xN+XcDiMSqViUq5TqFLUp9NpFAoF893c3BySySTi8Tjm5uaMEHZwOK6QSgG7dgGrVvkfHxwEmB01nV547+DQ53DucIcZhzMLj65mBwLdDG3pdNq4YZFQ6oovySFdgBj3AywQQiWLdKciYSchDAQChnyT0JFQaIxIo9EwrjzAgu8+Y2tI7FVMaNptYPFGhRRdJFzAgj8/95Wh5YdkPRqNGpc39gEJY7lc9li2bJcXii8mFrCFl9ZNY0RIJPnZhsYdKexyA4GAceFKJBI47bTTTLxRKpXC9PQ0AoGAER+tVgt79uzB9u3b0W63MTMzg5NOOgkTExMei5hm5iJBpSsQ5wFTGKfTaY8lsN1uY3Jy0sw7imTuDaTttmN3wuGwiVOhBYf90Gg0kMlkEA6HDUGmyGAadQAmRoRlMy0ys5XRzY59WKlUzGKCumtyDpJ8awIHdbkDupYGEnW679GFjO1lXFW5XEYmk1kkvJhWnNdyjrAcCiJaT0nomfWP1g3Grw0MDHisHZyrjLVau3atsWjQdY11pTssLbI67/g8se8Zj8Ox47zhnFm9erURgLr/Fctg39GqFAwGkc1mkU6nPW5yTPXO3wMKz2QyiWAwaH7jVLBx3jDukoKTx1gHdbmzM2TSgp5OpzE6OmrGjC63wWAQmUym52KUg0Nfo9nsbpr63e8C5567+HggAPCZ27Vr4fs3vxn4p386MnV0cHBYcRy0CDqWsvDYbiPcW0UzqFEAqfsWQaFCAkpCZO+rQeKjcRMkOHxfqVQ8hDoWixk3Hs34plYArkqzHuqXT8JFgquxNdoGdWXhaq8GOKs7nt1v/KvuUbZLIM9Tq5BaTLhiryvdhN/3fu5xdjyNWoTUCqaf169fj5GREZRKJTz33HN45plnjDVExw4AnnvuObP6PTExYdJOA97kEBq8HwgEDHEOBAJGcHC8arWaGSM7zmJwcNBjeWHd2Q9qIaJFR/uax+iapkKfq/csi+dS4HLuaFYx7TsVzWwnY146nQ4GBgY8gpUiHYDZx4pjri6cnAs8tmrVKk88DOcDLZ/ZbNZjJeQc5B5CrFO5XEYulzMihfEtSuBVTPN5aLVaRvDY1kqdd+oSyIUJFVscQ3UB1MQhnU7H7GFF8WFbNrUsLlLQvSydTnt+N9R1UC1K+vulAsieM5wDGp/Eea7jrxYuPxdEWtYoujudjhFs9jPs4HBcYHYWeP3ru+5x73kP8Ou/7n+eusV95jNAPg/83d8BPnHLDg7HPJw73MHhWMrCo+5n6s6mPvtKQO1jwOL9b9QdhW5RShwImwgoeQe81hw7gNnPlUxjUEh89BgJFF1rWAcVNLYg0TLUGmMHRWublBzr99o2WidUYPa6Xq/VeCW7T5V8+Y2TLeJyuRxSqZRxq5ubmzPuevZ95+fnEQqFsGPHDqxevRqrVq0ycSKaGlxFA0WbZphTEUTCrnE6vSxetLqoeORnHWvtS5LUfD5vRA3d1TjGTOQQCoXMvOCL7mNsi+5BZI9bKBQyz2fU8ovX9lAgaDZGW+SwbRRhurig4owWKX1+1R2UFhOey++1Lnyu9FlieRRbgDe5CUUIxZYeI3r9Rugc5/yiVUqfFz9LplrL7IUEnRv2tX7f6XOg8DvPb/FH5zjvq+fYc1BjC+3fDAeH4wpbt3b//q//1U2Y8F//69Lnr1vX/fv97y+r+DfhTfjP+M89j78Vb8Ve7F1WWb0wiEH8H/yfg77u9Xg9Wmj5HluHdfgMPoMWWng9Xv+S6udwjMGJoIPDZcdQFh4lUXT5Ghwc9GRz0vOAhYQFKjqUfNGSQAuOWktIgDudjvG1J1FTl6ZAIIB8Pu+JaymXyyYlsG4iqrEHbActQCrgarUaCoUChoeHjTsOyS3LYQwQsEBgNOCcBFD7gudpILW2G1iwmGl7bOHDv0q21G2OLlgcF90Lieey/2llUzGn96MlgEkuTj/9dASDQQwODuLZZ581m6PSEhQOhzE7O4vvfe972LlzJzZu3IgzzzwTq/b7f7NPuArPdMmhUAiDg4OGuNMSyLmgKYlZN7W8sX2aKZDWAptM072LLlOcZ9Vq1bixqQXIFtG2aNWx4LjyHHUD5DzgcY6z7rml40JXL44H5ytFlrqacs5znJjMQK1YLFetSnQJY58oWWdcDgWVJhbgPFHBZWNubg6hUAgjIyNmPDi/Nb239qlthWXfcKFDrT06n9i3FFxsE2OYotGoJwmFXwwXXWmHh4dN3Tj+On9U0GnCEX5W19RAYMHKSXFN65stCnVO87MTQg7HPZ58EvjsZ7vvDySEAOAXf7G7uSrRbgPvfKf5+Dv4HazHelyKS3ExLu5ZzCfwCRRQWPT9LGbxIXzIc14ccd8ykkjiWlx74DoLOujgj/HHaKON38fvYxLeTL1FFHEf7kMAAXwaixesv4Qv4dv49kHd83BhPdbjd/A7qKOO38XvHu3qOPTAgw8+iD/8wz/Eli1bsHv3bnzpS1/CG97wBnO80+ngQx/6EP78z/8cc3NzeMUrXoE77rgDp59+ujlnZmYG73jHO/DVr37VGFT++I//GOl0etn1OOwxQYcTStwZezMzM+MJZC6Xy5iamvJ8R/JEAkJBoi5KJGwUV8lk0hA8rrxrMgEGsZN8JRIJFItFQzjpRkQBRdHCYHBdjdUMTCR+4XAYw8PDxh2IFhCmDA4EAmbneroEkRiRDDLYudFoYGBgwGTVSiQSHpKu4oR9rBYBFU92imsNAmd7SKJsUanEWDfI7HQ6eOaZZ7B27VpkMplFLo0qECmGTj/9dJxyyin4xV/8RfzjP/4jdu7cabJvKcncvXs37r33Xnzve9/DL/zCL+Diiy82fagidGBgYNHKuLqA5fN5j6sSY6H0PJbHoHwVUCS05XLZuLJVq1Ukk0kzJxirQzGrZJzWDL63LWwqsnhcCT/PYXl6fS83KZvkt1ot5PN582zo3IhEIsjlch7STMLNPtCyWZ6Wz77URQZ+xznEfuM9VZzomLBMxiYCC/F+fKZ1XFqtFhKJBBKJhMeiq21k3/E5VkHMe9LaSGsiAGQyGVMf/o7ovNB2qrWX3/tZl9USBsDMOU3mwDLUKqXCSMWgtkEtWA4OJwweewz4x39cngg677zui2i3u1alTge48078SulXcAkOHPP8FrzF9/tZzHosRO/AO3qKIABoook7cMeB6y3YhE0IIogCCtiN3XgQD+JRPAoAmMEM/hR/iiCC+BQ+BQC4CTchie5v2hqswbnwiaU6CliP9XgH3oE66ngBL3iOfRlfxg7sODoVO0ZxtBIjlEolvOxlL8Nv/MZv4Lrrrlt0/BOf+AQ+/elP4+6778aGDRvwP/7H/8BVV12FJ554wrhp33jjjdi9ezfuv/9+NBoNvPWtb8XNN998UEaWvhdB/A+cqYEpcLjyqTEqwWDQBDhrTA7JEsvRfV0Ys8HzlYQCXtcSEjZNX0zCQjctjdchwQBghBBXznWFnxsWsh5KqEn8AJg4ACVPavGySTHLt+Nw7IQBDI7XeAUVIhQp7Htd0da4CttNR2NwNGYhEAiYTGiso10O62LHUEWjUQwPD6NUKhnrmZJIBu23221s3boVl156qYec8zymYdY69bK6aBZBJadqFVLLALCwnw8FqFrklJDqSj8Jts69XsTUtk5q37GvmPhAj3GDVpbv95f1orjnfKWwZx11TlFo8bmwLQu2dYFznARf+02fG7rL2X2iFhMViTpv9Hx1+9LveQ0XDHiu9qn+ltiCRcdTxSjfc8HDPl+fQy6c8NnSNqqQsa2LwEISFR0//r7Rmq17QbHveb4+g7ZVyMHhuMfevcADDwCvfvXBXRcMLliG8nl8/ytPYN/8voO+/QQm8G/wbzCIQfwJ/uSA589hDg/hIdRRx2/jtw/qXuuxHiGE8LL9/1JIYQ3WeM7poGPKHcIQXofXIYss3rj/H9FAA1/H1wEAV+NqhODNomqjgw6+hq8dVH2Xwj/hn8y9FQ/iQSeCbBwld7hrrrkG11xzjX9RnQ4+9alP4QMf+ABe//qu++Vf/dVfYWxsDF/+8pdxww034Mknn8R9992HH/3oR3j5y18OAPiTP/kTvPa1r8Uf/dEfLTvHQF+LIJsgRqNRTypc2+VMrSLqKkN3J33PV6VSAbAQcK7EyCb36o5CYkFCw5ViCitFLBYzyQ+YupZgSmi2w46PsONYlOQqkVRCBcBD+HhcLWEa3F0ul006Y5IhXb1WV0FuIMo+50q0knVaqmwxqaQ9l8sZUcD6+q2Kq9sjvx8ZGTEuVbSQabY9oCs6d+7ciUKh4EkpzH5mymYKT51zADz9o+LIdh2kGNE6axtoKeD80cQJtiUmGo16+lKFqLow2Sv3Nnml+5rOIUItAdo2rbOS72Qy6bEi6rl+rmr6bLAcvR9fOnd13yPbWsJ+4f14jcaf+bVF5xW/89tQVK0mPFcXIthGdZFTsAxtm9af4227wfF3hC56fI5sdzhbMGv71MLjN3fVZVJFplp2OeYqghwcThj8+MfAW9/atQgR55zTFTnLxd1349YbbgD++Z+7SRQOAtfgGnwCnwAABBHEOThn0Tk7sROzmAUAPIEnlow5Wgp2rM/78D78Af4AABBHHKfhNDTRxIW4EABwG27DRbgIWWQxiUnsw4LIK6CAX8YvAwC2YAui8N+DKYooTsWpeAJPmPMdHJ5//nlMTk7iiiuuMN/lcjlccskl2Lx5M2644QZs3rwZAwMDRgABwBVXXIFgMIgf/OAHeOMb3+hX9CL0tQiy4yLoFqcCQIO5AZgVXVqNpqamjHUoGOxmVaI1plKpGOLZbreN2xLvTXc5kglm7gJg9kxhvTRjnW6Aqq51JC661w6FXbvdRjKZNG51FBKRSAT1et1Ya7hinEqlPNYJZhmj2yCJP7CwozzvyTgUXb2fnZ1Fq9UybSoWiyaF88TEBCKRCPbt24eZmRkkk0kkEgnEYjHs3bvXuBeq5YOpoBknlU6nTXY3Clh+Zju074GFWCP92+l0sGbNGpx88skIh8PYuXMnHnvsMezatQt79+41FiegG6f1mc98BqtWrcLo6CgqlQpOOukkrF27Fuecc44n65i9pxDjuihk2B/tdtukKuZ46tiS8KsVhiSeVke1DFF8qqDkmPlZUJT0cn7SusM6JBIJI6wTiQTm5+cRDHb3omFGOg2GVxKv5QDeIHwKAgrKfD6PWCxmMutp/VQYKtlut9soFose6xnH3LbAkahXq1XPd3wm1VKoFjuKF1v8cY7qXGN94/G4EX6xWAwzMzMol8uoVqvI5XLmXC1PBYq6obGump5f78e6ajwfAJOdrdlsmoUKnQf2uKg1Tvet4v5dfMb5e6kimnNM+9VZgBxOSDz/PHD++Qufd+4EmOgpnV54vxS+8AXg//l/gHvuAfb/9i4H/7z/H9AVIn5WjPfgPbgHLz3G2sZt+/8BwJk4Ew/hIQDAA3jAc940pvExfAyfwWd8y7kIF/W8x8k4Gd/Gt3EBLlihWjscNFbYEpS3hP5S+3z2wuRkNyZtbGzM8/3Y2Jg5Njk5idHRUc/xcDiMoaEhc85y0NciSEmHBqJTzNBCpJaCUKi7SSL3Z2EKYCUM/A+fgcu264reh2WTPLFOJJqamYtEiOSHZIdii24q6n6lK/AsX1dquXpN4aSiAFjIBEfRRqvN4OCgSSPNzTEBGJJMa5S6q5EEazt1r6VEImEEQCCwsG8T9yDhd0CX8DNgnGWxn3gvClYVDiT1miiAq/hsfz6fN+Q3m83izDPPNJnkZmdnPWKm0Whg9+7dmJycRCQSwe7du7F161Zs374dp59+OtLpNGKxmEc0sl9oDbNTZWu/s894riY1YLto9SHJVlFvJ6SwN5+1oWSflk/eh3Wi9YsCNBQKoVQqmTb1Eld8Dgh1I9U5x3rw+VMBZFtmVADx2dJYJy2P52kd6LZIqEjkggHbbYsbtUDR4kTXNvahlqdW1FQqZRKwzM/Pmw1kddwAGFc9fanLWi8XPButVss8MzqmLE+Tm/D3j/W2x5GLDJryn31gJ/lQOCuQgwOAk05aeP+3fwtcdx0QWtrdCwDwF38BTEwA//N/dmOGDhJVVDGCkYO+biXwFJ46LPd+AS9gAzaseLkOy8dKxwStXbvW8/2HPvQhfPjDH37pNzhM6GsRRPA/cq4wq5BRl6ROp+NZbSfhsckYCYAeo/vQgUic7Wqj91YiZGfessvh+3a7bYSYxjOomxAJjW2pUrKvrlDciJOEh8kfeG/NWMeXHfei7oW8D5MUsF0qyNiXPN8WjXbbbdHBsWXZagW0XZB0JTudTi8i+nQ95Bjxxb2dqtWqIZ0TExMYHR3FyMiIZ6Xcro+KQ1rj2CauxFerVRQKBcTjcSN6dQWeIpn9oHFbbKdffJEf6N7I+aLiy3aNVMupCqhAoOtKyLaxPJ1PWmfdmwqAZwHCtlDYgkgJ9lKpmP1Eggb387O2xRYzuqCgvx1sj/5lP+siiZZju6LpwgUAE8ejVhX2NeCNfVL3N60H70OLl66q2fflgoBaT+3fNhXodt0J+ze0lzhzcDih8Z/+Uzed9nvfu7zzP/IR4JRTui52Dg7HIbZv345sNms+H8oG2+Pj4wCAPXv2eJIZ7dmzBxdeeKE5Z2pqynNds9nEzMyMuX456GsRpGRQA5+VVCgB5Cr40NCQJ1W1vcKpBI+EkKunGvCvWZRUdHAVVt3NlEjxO8IvWJtgIL+dlpckp9PpbqZaLBYxODjoIagaE0Ghw9dzzz2HVatWedKJk9jRxU5Fji2qNABe3XJIltQtCliwYNj74mgcko4hV9AZQ2VbFNjndOVTSxjHOxaLmU05s9ks4vE46vU6pqamUCwWEQgEPPvGsL+LxSLK5TKef/55nHvuuTj77LMxMDDgERKhUMizGWgoFEIikUA4HMbU1JTZPJcxauVyGXNzc9i9ezey2SwGBweRy+U8m1qyH9XCxc/sC43j0PrQ0sdxoLtbu9022eZo4dNy9Rx1d+SGsoVCwbhysm/50rEMBrsbBuszQVHOenJu83yd4zqutvVCn0taaFkW+0PLpDWRbqc6Z2ntsd36KOL0eVZhYT+XCjsmTa1izEapViBaYPiMMGNjJBIx+1Ip+NyUSiVjcdX+skUKs0TabofqCqj10bgm/c1gX6r497MQOTic0Pjv/x144QXgjmVkYwsEgDe9Cfi1X1t8bGgI2J8S38HhiGGF3eGy2axHBB0KNmzYgPHxcXzzm980oiefz+MHP/gB3v72twMANm7ciLm5OWzZsgUXXdR1ufzWt76FdruNSy45cCZGoq9FkAYu63/oJLUkdyRDwWAQQ0NDhgDRN15XPIGFAGtdLSZJ5/1UZCmhocuWWlJoKQAWB7MDCwHx9io5yQ8Fgb1yy3vR8qAr3yxLM+HxXp1OBwMDA8ZKwtVnkuVsNmvOI5nX+moigXq9btyOFFpHpgznOABdosuy7dgGtYZo3XXlnffQpAzabs3Y1Wq1EI/HcfLJJyOdTuNHP/oRtm3bhkQigb1796JarZq4JbusZ555Bvv27UOz2cTpp59u2qqb6TabTWzfvt306fz8PObn51Gr1cx+LHNzc8Yl8LnnnkO73UYqlcKrX/1qlMtlRCIRxONxE2cVCASMdY9t4L0Zt8XNc2ndo9thrVbDT3/6U0+K6L17u+lVo9EohoaGMDw8jEQigWg0agQQz+t0Okgmk8hmsxgdHTXPC/e+0vk2PT1tROhFF11knjcVzrSCpdNpNBoNFAoFJBIJj0WWz7NaYTnGXICwLVC2VVBdTWnJo9sknz8+I5pVkWXSHbXRaBgRyDaoWx3jcXh/zm2Nu2EsIn9fOGa1Wg3lctn8FukqmSYUYV3ZVtaXMYJcpNA+UutUrVYzv3m0hHY6Hc91HEtuJpxKpZBIJExfsQy1ZKkVy8HBAUCrBdx1F/AP/wBs2AA8/PDS54dC/u5zzz3XTakNAO94B/DFL654VR0cbBytFNnFYhHPPvus+fz888/jkUcewdDQENatW4d3vvOd+PjHP47TTz/dpMiemJgwewmdffbZuPrqq/G2t70Nn/3sZ9FoNHDLLbfghhtuWHZmOKDPRZANki0SD27YmE6nF7l82Kuo+gIWEh/wXBIhftZ4IIogpl7W80lmeS8Ai1bR7TJt9y5dSVeSRHKsWeNI2hjboW1SEsmsXuyzmZkZU06pVFoUl6Grv7rarfui6Oo5BSEFHDef5fd08dPVfF0ZV3FLywatWPZKtY4LXQRJGLXOsVjMZBPZsGEDnn32WQSDQSMedu7cucgliDEfP/7xj/Hiiy+a/l29ejVyuZxxbbv//vtx2mmnYe3atXj66acN8dRYkkwmg/HxceRyOeTzeVSrVTzxxBMmjTf3WDr99NMxOjqKcDiMYrFoLH+jo6PYuXMnSqWSx8IZiUQwNDSEZrOJQqGA+fl5DA0NGQIbj8dNgGG73U08kM/nPckuaDnKZDLIZrPYsWMH5ubmkE6ncfLJJ2NwcNCs7tAqxzGi8GS2vXK5jHK5jNHRUTM+Q0NDHrHCea7PAEHhxPck4xT1GjdnW43YN4x14/iokOKcz2azi9zDKGy50SvnViKRMOcxtbTtZsb6UOi1Wi3PJqf8TchkMh63Ri4E8DfHdrek8KKQKpVKnmMam8Yy1K2Ofch5qP3H/ZxYDx7n7wufAe5VxvY6ODgIqtXua24OuPDCbva4LVu6lp/lQoO8//APgfe/v5tR7v3vX+naOjgcdfzrv/4rfumXfsl8vvXWWwEAb37zm3HXXXfhPe95D0qlEm6++WbMzc3h3/27f4f77rvPkz3585//PG655Ra85jWvQTDY3Sz107pp8TLQ1yJIibm6wKhliP/h20JHCYK+VHBQDLAsEgt1PVHCT2LPjHIsx28vkF7t4L30vVqOlICwnRQvar2yffjVFY915MowSTE3vSQ5YruUsOmqM1fe+b2SQr5n3AuJmfazQsfDL1OV9rUtYHWzW5JitQpoWdFoFBMTE8jlcpiZmTHuSolEArt37/bciwSfAmlqasqshOfzeYyMjCCTySCVSmHbtm3mftu2bUMul0M6nTZWtXg8jnQ6jbGxMeOKOT09jUKhgKmpKZNCPRgMevYOqlQqhgBHo1FMTk6a7H7MKgjAXM/zTznlFDNWyWTS9EOj0cCePXs8sU/ctLbT6Zi+ALorNbSCnXTSSR5rBPs6nU6beTg1NYVyuYxSqYR8Pu8RLLS+0BWMGQ35XKk4sl3eKpWK5znLZDIeoQ/AJHqoVqvmWZyfnzep8ylOKBr4rNDyQ2uWzlW6SqpYV+uvZs7jIgbbzDrQgsz5x/5TwaPPuLqmse+ABQsMBZ79+6Xtst0P9XkLhUKejZTpsqnPHcvX/bP0d8CJIAeHHqjXgZ/+tPt+/4o1Pvc5YOQgkwqsX999jYx0rUbLjTlycDhYHKV9gi677LIlXasDgQA++tGP4qMf/WjPc4aGhg5qY1Q/HDciiMRDA7tJOpTMqJ+8+vqraxNJ44svvojx8XEjhDSlLUkLScnMzMwi8sbzuSJtp7MFFki+En/CFjEkbVoGyUosFvNkXGMdlOywX1ivWq2GYrGIXbt2IZfLIRAImHgCljczM4NAoLtvTyQSQblcNkpcCZEGdtMVjeSY5IwbxgILWfrUdYjkzN6Utt1uI5fLGfc/zXbGdMHctJMEkPXSmBs9v9FoYM2aNXjxxRfNuKnwscvR+gcCATzzzDN49tlnTdrviYkJ5PN5PPLII8hms8bla2hoCLlczqQMj8ViGB4eNqQ0m80imUxi586deP7551EulzE7O2uIKTOttNttI4AikQhGR0dx8sknm4x3W7ZsQaPRwKpVq3DyySdjzZo1HvFHq2G73cbY2JjH9QxYyP7H+LWxsTGcffbZePjhh/Hkk09iz549aLfbGBkZwcDAgJkfFMPMOJjNZs282rt3L+bm5lAoFPDII4/g4osvNjFHtD5yzyy65tHCNDExYRI2lMtlhEIhzM3N4Wc/+xkuueQSVKtVzM/PY2ZmBkA3ffTIyIiJcSuXy9ixY4exztK6wYyD0WgUs7Ozxj1tZmYGa9asQavVQr1ex4YNG7Br1y7jsjk4OGieC1qWKWj27NmDgYEBYx1KJBJIp9Not9vYt28fpqenEYlEMDAwgIGBAbP4QHfGSqXiSXFPq2WlUjFxZlzc4bPO54BimM8P55Xub6Yublw00PFWqxYtaPpMa6yeZqZzcHBYAtxb6L//dyCTWfrcP/xDf6vRmjXAf/kv3n2J3vOeBbc5B4eXiqMkgo4V9PX/ZkpSgQVSXCgUDKEnseXeJ4wVsONkAO8eHZFIBBs2bPDErDCRAglks9lEqVQyZIarqCSGFDrRaBT79u1DKpVCZv+PIV2+SGiUlKirHbCQ/lstWCxDhRNJFK9RS5JaSTRVdiaTwYYNG4wVgLE6FChMnkByqKl1KR41m1en091clYJIrWzcfJSr9iOyOkYhUiwWMTc3Z4QoV/K1f9TiFAqFPMHkJIEqyjiu3FeJ1w8MDODiiy82MTy/8Au/YEQtY264ms/c92pRUxcnxgyddtppWLdunUlEkMvlUCqVzD250j84OGjSLJN8V6tVvPjiiygWi2YeTU1NYX5+HpVKBZlMBieddBIqlQqeeOIJQ5xDoRAuvPBCDA0NmT2aaNngvLGfF51fFKpAV4zw+WDM0mWXXYZ9+/bh2WefxWOPPWaIei6Xw8TEBDKZDNLpNMbHx006cT4vdMNLJpP48Y9/jGKxaGK0BgcHMTQ0hPHxcYTDYWzfvh2zs7M45ZRT8N3vfheFQgGNRgPJZBJDQ0MYHR3Ff/gP/8FYLVetWmXmDJ+NJ554wrQzFoshn897Fj9GR0cxODiIkZERzMzMGNe0vXv3muQXzWYTg4ODGB4eNgkvisWi6bdQKIRnn30WuVwOa9euNfGFc3NzJuEIn0e6ErIOU1NTZs5wQ95wOGwWIXSMaOnT1Px8xrjYwt8De2NfTXFPocnnbmBgwMSrcRGF5XY6HaTTafMbwu/9Pjs4OCwDf/ZnBz7n1FO9ny+/HDjzzO77iQngd3+3+77TWYgduueeg9581cHBwYu+FkF+LmkUPnTtst2qNIOYbnZpx9mQgBN04SFoSVE3ItZhfn7exMJwpZVZqfysT3aaWn2p65wmb2AZtrXIdgFTEaIB/2pt4R46XOW1Y5ZYb7aF/aJCSON3NH6JK8nsTxWVFIpa11KphJ07dyKTyWBkZASJRMKIPRL1XC7nyfil/arjZ7sEsW48l6v1XF0fHR1FvV43mdzm5+fNij1dx/wIoK6+01Km2ex0DCggOQ8BIJVKIRgMmkQTvBc39GV67mQyiXw+j0KhYMZrYGAAyWQSqVQK2WzWE1vCPtOYK62PH1RoBwIBpFIpIzDr9TrGxsYwPz9vNrule1WpVML27dsN0abAU6FO98FOp2MyyFC0RaNRjI6OmuD8tWvXolwuG/c3zu3Z2VkUi0XkcjmTKCIajaJWq6FSqWBgYMDTnnQ67dlbKpPJmPuxjqFQCMPDw8aKyueD84lCu16vo93u7qUzMjKCdruN3bt3o1gsmgWEwcFBj5seRT/nqr03lu3eaicB0bmt1mM+dxwv/S3QZ8X+/dBnlQsKGpuldVNrLD87ODgcBvzX/+r9/O53A//+3y98jseBV76yay36zP5NSctlYNeuxWXV68CDDx6+ujocVwjsf61EOf2IvhZBSi75Hz/d4tLptMcKoEG+JCVqXVCrkMYF2ERdrTK0LgEwcRW1Ws2sYFNMMPOXklN1p9MYIPvVS9AQKmzYFgAeaw2vVeJF9xlgIW6H/UD3GlpPWD+SQSae0OB1jUfitdxslnWgMCS5s+MzWq3uxrF0U0qlUgiFQkbYksBls1mP+yLHSN2+WF+7zzThAkUKXdqYIKBQKBhBWKlUUCqVUCwWPdZEFVO8H1Ngn3LKKcbFqFKpGKsfV+U1xsuOF6M4oBsTiXEsFsP09DR+/vOfm2Qfw8PDGBsbMyKBcVysH8efn9U6aItudYtTl0AeS6fTZvPYmZkZk6hi79695pnbvXu3sS7GYjEjpPksTExMmPEfHh4284RWq7GxMTOmZ511lmlLKpXC7Ows8vm82Qk6lUqZ/mMZ1WoVq1ev9qSgHhkZMYkaKDYp3Bh71el0kEqljFWTFlV1GVRBG4lEsGbNGuzduxc7duwwcziZTOKkk07C1NSUsfJyvvBF4cZFFLU+2xYZtQrz94pzhwsyGk/E8dLnUoUM78NnieDeaWot5rzWhSDNVufg4HAY8Yd/2H0R4+PAd77TfX/GGV0xdPfd/tdOTXUFE/Hss4e0OavDCQLnDte/oJ8//3Om2Mjn8yZ+g/70mgEqFouZGJ3Z2VkT95JIJAy5arVaJnsWy41EIiZImwHkJLalUskQslQqZYLZmaEulUoBgLk/3YU0XoUgYSF5VHJCNyl1TSHBAhZilWhF0NTPmlSg0WgscmOjUGKSBU32wHpqSnGWp0KLgkVJOV3KGN+g9ySBpgAbHx/H2NgYpqamkE6nTVvL5TKy2ayxJNDKxBgmJXipVMoT48OYkEqlYtzsVBCmUimk02mTMSyXy2HNmjWYn5/Hc889hx07dmBkZMTEpxQKBWM95HfVahWVSgXT09P45je/iVe/+tXIZDKoVqsekV4qlZBOp83cAIAXXnjB9BctLxyvTCZjUi3HYjFs2LDBI555Hp8DdbtTt6tyuWwEGMcpFOrux6Sb7PKZsVMhk5CXy2Ukk0njMjUwMGBIfT6fN7FMdPOihbRerxviTiHGeC26mRK2qC0UCuaeExMTqFQqqFQqmJqaMlbMeDxustdxHiQSCRQKBQSDQWMBYgpvJpdgG7kooDE3+mzm83nzO8AFgIGBAWQyGePqSnGVSqVM3+fzeSNoOGf4exQOh026bCad4Ia+fHb0uadFi/FUbC/3EeOY02ql48byOQf5PcvjM5HNZj1xcPrMaip1BweHI4jJSeCss7rv9+3zxgjZiES8abo3bADm5w9v/Rwc+hR9LYLo3uNnDSEhqdVqGBgYMKvK9XodlUrFrLxS9JB0MZ6IYoDubtVq1RBOigC661DQADBkj8JDV2jVv5+ruRRZag3idepWpsQQgGdFV137WA4Ji8YZaLlKmrjCTFKu6bYpergirZmnSI60T9R9jmm3WaZaJ9QSwfFibAXjRhivQDc6uhLZq9m2yw8JPL+naKJ1kMc0+xbrYs+jXC5nBE6tVluUlY1lRSIRk0hidnYWmzdvxsjICIaGhpBKpTwWQ1oZWFfem4KZiSJCoZCJw2IcF+cAhYXuD6NWTY6bWknYp4lEwvQfybIm3NB4IZ1vFM66wbDGyTGjGrCQhYzkXa0SfE7UvYvPAUm/zt1arWbmNwDP/fm8Mf6pXq+bOL16vW4Sm6hLJwk/xQitl5zPXFTgs6/WEc57lsFnU8eR481FBI2bY0Y8xoBR6PG3h268vEatWsxiRyu1un3G43Gz4KHJSzqdboyeuuexjexLPtMcP47x3NwcUqkUUqmUSeCgCwgODg5HAQebac7BYQkcrX2CjhX0tQhS64VaJ0gSSbLoksXv1E1IY19skUJxosJCibeWRWJju5+Q5KsA0H04WJYSUF0VV9KvxEXboMKCJI+EieKFgkYFo/r/EyocWHct344PsN3zWGftI56nYoXXsWyKSWBBfKl7HUWm3Td0sWPfKIHX9qgI1PFWS5a+B2CEEwkrhVClUjGbj4ZCIRNbRNTrdezcuRP5fN4keSCJpyVKkwdks9lFGe/Uwsc6qXDjGKuboAo4jRuxE2jwOrVQsX5qoWJ/67OjIpJWTdaHc1fdTNUap+nU2S4V4TqvaIGkwOD46nxjuXRdVGFIEaBprPWlApR9on2mIojH+KzRfZRlFYtFpNNpT9INvZeKN44vX7ScqTVP56MKRVpUOQfUWqXPKsfGFrW2azC/s+utG7+ynvw9cXBwcHA4juDc4foXJFO6SzxJMolfOBzG/H5TsBJhHteUr/V63axgsyxd4eUqKcliJpMxq710h2MsgJJUutKRnLActUiQ9PMzCRSzdbHOjFdQYaeWHcYV0C2NZZEIkfAwYxRdsHgPO3heBSX7C1gg60rSlGwC3UxjtL7V63XjBgjAfM+yarUastksYrGYcR1k2bQUcUwYa8FEDfxeLVFq1dEYF507KgBtgUdyODAwYLL6FQoFk1K5WCyiVquZIPlisWj6qVwuY35+HtPT03jhhRc8c5b9m06nkcvlPMkANHkHybC2A1hwAVUBo4KVhJXnc2NQxqS1Wi3Mzc0Zy5yOtxJ09gUtUSom1aWS6ZcpVOkeOD8/j+HhYeOaSCsSkz9wXFWk6X24gSzjoXiM812TeDDtNF3+BgcHTTp3znXdrFetlmwLLVg6jzlP6TpI0UqLCft69+7dWL16tXGD03HRZ5+xT/yNYsIUIhKJmL6g5Yf363Q65vepUqmYutPNju1hvxSLRfNbpPt/2UlDaAFjPCNTc3c6HdOvtJqp8HNwcHBwcOh39LUIImFRQUHrAF3hSqWSCRwHFoJ9SQKZNpmuWCQMJCq6z0w2mzVuRo1Gw6Qy1vqQtA4ODhqBxKxh3JS0XC5j9erVHoJE8kPSQ9RqNU9mMrXoaGY6ElwKrHQ6jbm5OQ/5IZGhC46KP7ZDV70JXW3WFODcnFKJ+FNPPYVIJIJEIoHh4WHThmKxaGJPODYUOwBMTEWnsxAwzj61STLHlxYCv7gkWpAoloCFzIB0MYrH48b9UVfHCV1RTyaTJh6JY79r1y6Ew2Fs2LABoVAIk5OT2Lt3r7kHiaYKFxLNQqGAfD6PnTt34plnnsHw8DDWrVuH008/3WMZ0H2XOD84Vzk2BOcmiTvnSyaTwdTUlEnYsWrVKiOGbbc8tpv1Zf9qZj8l4LyPkvRUKmXENWNKOG95T5anVkbWm9Ydolwue4QpXSQpADjWg4OD5ppWq+WJbVOrCdto10nr0ul0zP491WoVo6OjZg6qZUznlKYl12eOCwDpdNqIQWDBEqz1oljlc8P7MfV4rVbD3r17MT8/j8HBQROXxGe2Xq9jamrKk4yFiwy2dY73t5MhaAa7RCLhWVSwLa0ODg4ODn2OE3hdq69FEGHvzaOkh1m/dKVbs0JxFbRYLGJqagqrV69GIpFAPB5Hs9k0wcckSFwlBbzkgbERJIcauE6SwlVtBj9zdZrChORJV+eZYEFdv4hYLGaC8qvVKnbv3o3h4WGTUCASiRhyxv5hf6nrDwOj2U7ua8OXuudQOCpxZVmNRgNDQ0PGKkByzXGoVCqeGAwln7pZKlfvmdxACT3vTxLLflKXH40L4mcKLz1PBZzuxWK7YamLEonzxMQEBgcHjcDjCnw2m0WpVDJxN5wXvAf7QOtKkt9qtTA6OmqsQhSnzDCoc0OFnc5jFaw8xsD5bDZr+orxKBpvxfbZlj8VRYyNUxc3XWBQ6ySfFfYBx4AbkNJSyNg6EnMuBHDuaeKMZDJp+oTzg3O4Wq0ilUphbm7OJEvgJrUUk7QaqrsXBZyfNZOJEti/jUbDLGTQejI9PY1Wq4VisYhAoLt/EoV2NptFsVhEqVQyFkzOO7pSsm8plunOx/I5N7dt24ZgMGgyzO3ZswczMzPYt28f2u2FJAY6l9VFkZYhfa51cYTjyd9HbszLPimVSua3y8HBwcGh/+FigvoY6rJix6GQCNCNhSvJ6j6nwce02jDbGAkT3VW4Ss1AZztDm5JPYHF2LmBhHxC6tLB+JOe2D7+2g5m2tGxN51uv101gdTQaNQRRA7Tt+AQVIawfBYjWRy0iGsOgLkX8jnvH0H1P3fFKpZJpsx2noUH/dgIKugHyXhrET6grG4ULy1MRRJFKkcP7q6uVWvTU2sT2AgvubOVyGdu2bTMWKl6vVjoVQRRfnLtMjlEqlVCv17F7924MDAwYKwbbrTEvtsWKoFjgfOccogimkLJdvFREan9QwPAadYEkudZNOW13KY2r4fV8xpRcqxWFgk9FOxNScF7r887xZca3ZDKJarVqRApFOeekxrhwPNVCx7lki159/hmXwyQedLelAFHXMWa3428G56HGFvEZZV9oTBMTPNTrdezZs8cIp3Q6bTaJnZ6eRqPRMPsu0f2Q40jhEgqFjMDmOFFo6XjncjkAwMzMjOd5KJVKnkUgBwcHBweHfkbfiyD+500yYQeYq5UFgCFcvAboumJxV3q6GVEYqSDQ/YXUPYwk0xYpJFOMgWB8Ba1D9Nefm5szpDSZTHo2V+Tqba901iMjI6hUKsjn81i3bp0hetxvh8Ru69atiMfjGBgYwOjoqEkRTNLJlWklldrPbLOu0JNEsh9t1zSKILoEFgoFpNNpBINBzM3NYc2aNYbwFotFj/uOJjxQss2VdHVj0pTO7XbbxEyogOXcoIhlX3I/HxJ0zTzGtlA4VatVI77m5+dN3U8++WTTZxzfF154wVh9WEeOI1f5SUA5X+v1Oh566CHkcjkMDAxgYmICExMTSCaTxqIwNzdnXPlYf+5nZLsQcjzoRlUul9Fut81+OZybmgxEBaBaSPhZY7poweF9uTkqY1NoDWN/apICzgkS83g8bqxpLI/1YVpqWnwYt5LP501SCW6EOjs7a0T91NSU2UyWc3xubs6IaHUXBbqb8NKlkM9xs9lEsVhEtVpFqVRCo9FAPB5HJpNBsVjE5OSkESPBYDeF+N69e42g2717t7FEV6tVzM7Omn5MJpN47rnnjABeu3atiTtrNBpG0FA4z8/Po9lsYu/evZ54Igq6qakpz/xi21588UVPLBWt4ZFIBKVSybjwATCp7DudjunLdDqNZDJp4uB6iXAHBwcHhz6DS4zQv9A4BnVxCQQCxtJCYkW3klAoZPb8UHcxtdLQh56ptFnmrl27jLscXdlITCuViom1oOhSUm6vbrfbbczPz+PFF180Li4q4ACY1XsVGSS9wMIeSBqcT+IYDAbN3jbNZhPDw8MmlXAymcSePXuMJYnigEROY2Qoqlhvklz2N/tYySSz6LEv6GJD6xSvmZ2d9eyBo1nAtLxIJIJ8Pg9gwSVL3do0e5xuQEkhp+KNrnXsz0qlYvqAblq8D+cFLSaFQsHEiVQqFQwODpq9mNjvJJTr1683At0mjrTalUolBINB7Nu3D8ViEcViEQAwPz+PQqGAPXv24Kc//SkGBgYwPDyMTCZjYm2Yvp3jQUsJV+vZN7RkUKTTPU7TV9M9DoDZMJbnDA0NGetHrVYz85Tl8F60fuVyOWMRi8fjHivc/Py8EWcUBnxVq1XkcjkjkjUb4PPPP+/ZY4fiCQDy+bzHLW5oaMhYbMrlMrZs2WKeO42/q1Qqnt8LPq+c+xS1FMFq6dKFAAoaTWWdyWQ8brlqOeE5+jtFV7/du3d7UofbFineg3VgeYxz1Lmu2fU4r9V6Ojc353F5ZX39LNFcMOFCg4ODg4PD8QHnDtfHsF3d+J823Ue4csyAYfrNa9yHrm6rSxQ/M9g7Go0im82aVXYSAxISBoOTnGgAsZIPki2SMlo/bAuLxlYoWeT3LNeOS1LxovuZMIhcg9QpXgiWb7vmkdirOx3B7ykW6SalViGWYe/LotYngi5UtqubxqWQdLIcte6olU7HVNuo1gw7VbPG0/BFi5zGZ+mmmbxO26ZZv3gu+4vB5/F4HMlk0lgD9+7di3w+b+rFWKFms2k23KWFZWZmxlhpSKLZ93bsE+evun5p2mNeR/c9ih5aTvR8ziEApl6cK/V6HXNzc545xv5j3B3HgqRdn7O5uTljmeNz3Wq1sHfvXmP54/PNec6NPwlmbeMrn8+b+tmZCXU+0+pou3fymFq0dH6xXHV/1aQPFFs6l/V6dUu05yr7Rp8zzmH+RtiuqTrHtF0qxOxnS0We/lbpb6qdKMXBwcHBwaHf0fciiGRaUy6XSiVPLAw/k0hQHABdQqGuV0p0SEDpYpJOp42FqFqtmr1BMpkMhoeHPfugKAlVEs/jTO+s/vs8RkJCkUACwvqwnhQnStRt1z+SQbqQkfDSEsbPSmxU8JDUK8nXFXON31HyqGKG5+lKM60vfjFbjJlRtz+6f3HcSfRpwdEYGAorlqeuchrjwPqoayTvyWuU/NKiw4xZ7FsVrRxzFaNqAdOMXLFYDLlcDrlczsRfcMwYg0LxVavVkM/nTf2ZTpluVxwX1lc309Q6cGzs7Hocf+13oOtSpuJVxTEXBFT8zszMLHKrU0HJ+aVg3efn5z3zUOuxFPTZmpmZOeD5S8HvfvZvAp9v7U/tP7pvhsNhIxL1GWGb+ddP/BBarn6nLpsamwh4LeT6W+TXRn3vd55aRh0cHBwcjjM4d7j+BUleq9XCE088YchhJpMxK+WdTgfJZNK44SQSCRNYnMlkzIo8ySQJP8k4ffNTqRT27NmDsbExjI2NGTJE0qrxQMygBXj3X+GqPeBNvR2JRIzFinXVdNbq+kKhB3QJH1eSKaZYJt3LNA026xYIBEz8BgAPqWV9KpXKos0RdaWZ7yuViom5CAa7u97TvY9B7BREHDPGwjDzHq0XLJuuUiSIdG/i2MzPzxtiTssJ41wKhYKnXN34kWOk7nqDg4NmfHK5nLEezszMYHp62iMyCoWCCc4nqeUYc7zYt7RAMOZFrW4UUszmRevOwMAA1q9fj1KphEKhgKmpKTMHGctDAUOhqOmjWRfOFVvcqIhRy51a2fQvsCD8WI5aNSjm1fKg/eIHPytCL1J+LEIXHRS6CEDXxkAgYGKM7Jdep3BWFgcHBweHIwXnDtfHIGlvNpuGfDNgmESxUqmYuIVOp4O5uTljBaFLDq0RdE8j+S6VSkZEZTIZszcH3ZvUZW1oaMizaloqlYx1iGQ6FosZUq0ucYz1oXWCViKWlc/nTdYopscFFuJh+Fnd4bgizfPogqWryOzDUCiEcrns2Y+Gq8B+7nJq1WDaZYqAZDLpsQCwTaVSyVjP2FZ1u6OlgmPa6SxkJmMAN93MNEkC78u253I5jysQV9nL5TJ2795txo/WJMaacC5QECQSCQwODnrGee3ateY+FK202LAPKLaZDIFxMdpnKoxZf2YbozBn4DqTXNDaRcERi8U88U/sAz83K44r+4zCUa1e6gbGvtN+WYrA21YMtXLY7mZ6nT2vbMGmZR0IfuJppQSF9gmfOVsM2ueriF8KTvQ4ODg4ODgcHRwXIigQCHisIEz5SsuJrtIy3kFJn8YMMJaDpJYkfGZmxmSksglzKBTyZJpirAKJ+8zMjBFATNRAQqtJDgKBgEm4oC5bs7OzRgQxpokvTR1NixCJMgVVMBj07F1DKwgTCiQSCSMOKRDYLvazbfmgKxhFCcmcxlexzjxmu+mo+5YKIm0Px0gFHs9VdzzWk/swqYuWihJNUFGv11EsFk0b6NrGumv6Z4oKlqFpuimGKIg0rTn/anC+uttp3BHdFCk+6R5HgU8hS2uVClYmyGDf2m5udr/b7/nZ7xmzLUXEcgk856mflcgWVUdTFPi5nh2KuNLnxLbAOTg4ODg4HDNw7nD9Cw0KJ+Gg1YGEW91XbF9+G+qWpaAlCYDJ4GVj+/btB6yvHwmyYzF6ESYl9STydEfSOJt0Om3iQbhqHQgETEY1Wh0Y45TJZDA0NITJyUlj8YrFYh7rGK1V7L9oNIrp6WlMTU0hFArhzDPPRDabNXEq6sbGMaCbH0VJvV5HMpk04oECkW2lWODY0aJCSwvjXtQNkW2mhScSiaBcLqNcLiMajeKMM85AJBIxMTadTge7du0yacNpDaM73Pj4uIkXm5+fx/T0NFKpFIaGhjxCLR6Po1qtmjTUFD+BQMC4q1EAUFipCGfcly3wmJaY85LjQ5crxiXRbdIWVHSj05glPxHSy0WN/XyoGcHUgsJsdrYbmZ/F51DEwksVGHx+NEMhYVupVOT4Pat2bI2Dg4ODg8MxCSeC+hc2oaOLEQmunZ7Wb6X3cNXLvsdyiJEfkVNCxjIoLjTuiAKFGzdqkL6fm5G6N7FcCiomZCApp6saX3Qho9h58cUXjTWFFgy+dG8dWqMo5iiUms0mUqkU1q1bZ9Ik06LDzSiZgS+VShkXvE6nY5IHsL9pnQJg9iZiPBZdytQKNzExgU6nuydKPp83boq5XM6zsWQ6nTZ9yjKZiQwApqenzR4vahkDYPZvoaVt1apVxnrJZAdMFPHCCy94Ntvcu3cvyuWy2WCUokYTZHCu+1l1bJJuu0MeCVBQHa37Lwd2vJ0f1H1Q+9fBwcHBwcGh/9D3IkgJFckhY1GUJOo5R8Lt5kAB4L2yLi0l1DQNrrpU6XH73rYIssvjS609KozooqfXck8ZWjFolWBMCl8UOyyHLlvqXkdrBfdcYpIKxuzUajVjxVKXQo2FsYm+utEVi0XjjsYYHR6jW6Rm59IMcPF43CQvUDdKWtc0a9r8/LwRdJyH6qqm7n7btm0z/cu9gvhS6w1TLHOcmQxBrTt+c2O5bl3LwaFagfS+7I9jCQcjYg5kwV1uOQ4ODg4ODscSXGKEPoZaJ5QcAov31bBJy+EUQsspt1cg+IHIqk18bWFkE2ES716xH1pXdcnTvzaB1Tgbuq1pWfxsW79oZdG6aRtoFSE0TbWKK3Vr1A0vNX6I92g0GkgkEka00IrFdmibE4mEEXgUWrRKMc6MYk/nF93e1GWQ53c6HVQqFY9o0/g13ROJded7zfBmx+b0GkPtWz+L5MGg1z0OpRwdJ51b9n1e6r1WGssRQA4ODg4ODn0J5w7Xv+AKM1MeL7UpoBLWI2EJOhBsgnqwhIr11+xt9jFg6ZV8nscYFv3uQP3jd82BYFu/OA7aBnsM7fJ5rNeeMwA81idgwXKlx+zyABixwhgqAJidne3ZDr/vaXGiwGOMGmOsABi3NlrSetVnqXHs1ecrJVx4/Uo+KxSOFJyMEbPvcaQstSslYg61rk5MOTg4ODg4HD30tQjiCrNuNrocHG0BtFJYiXYcSkD6gc47mHr1Cjo/1Hv5CaaDmRvLOdcWr2qpsq2SupGmumn2qn+vtizn+8OBlbiXbQmKRCJIp9MmBovnHGmshEh8qfDLkOjg4ODg4HAkEOh0EFiB/39Xooyjgb4XQXY8iMORwVKk9UAr3PZ4HQ4C/FJE8cG4JNrfaxIAuvNlMhkT69MrjulEgIpS3fOKCStONPi5rzo4ODg4OBwxnODucH2dy5VE0s8l7EB4qfESDg6Eij6mp9Z4plwu59l/yS9V9bGOlXpeVAQmEgmzuS5xoolD/n4xmYeDg4ODg4PDkUFfW4KAhZ3ol5vFyna9WiqjlsPKwHYVO16h7dNkDVNTU55jhyLajxUwVTg3hj0UtNtt7Ny5c4Vr1p843p8JBwcHB4djFy47XB9DM8MRy83MxpTERzMmoZ+x3NTCRyuV8LGyqu4nro+Vuh0M1GLRbDZXPOV1P/aJg4ODg4NDX8O5w/UvGIB+sIHvmpXMka/DixO5f/1Eer+CabqdxdThWMRtt92Giy++GJlMBqOjo3jDG96Ap556ynNOtVrFpk2bMDw8jHQ6jeuvvx579uzxnLNt2zZce+21SCaTGB0dxbvf/e5jbo8rBwcHB4eVQV+LIM2stFyoCOqV6tjB4aVCBdDxIBiCwSDC4TCazaZ57+BwrOCBBx7Apk2b8PDDD+P+++9Ho9HAlVdeiVKpZM5517veha9+9av44he/iAceeAC7du3CddddZ463Wi1ce+21qNfr+P73v4+7774bd911Fz74wQ8ejSY5ODg4HHbQHW4lXv2IvmYyXJU+FDHD4OxeG306vHTYmdBc3/Y3Op0OCoWCSUkfi8WOdpUcHAAA9913n+fzXXfdhdHRUWzZsgWvetWrMD8/j8997nO45557cPnllwMA7rzzTpx99tl4+OGHcemll+LrX/86nnjiCXzjG9/A2NgYLrzwQnzsYx/De9/7Xnz4wx9GNBo9Gk1zcHBwOHxw7nD9j+WSaxU8oVDIbN6oe7s4on544Pq1v9Fut9FoNAAsbArr4HCsYn5+HgAwNDQEANiyZQsajQauuOIKc85ZZ52FdevWYfPmzQCAzZs34/zzz8fY2Jg556qrrkI+n8fjjz/ue59arYZ8Pu95OTg4ODj0B/peBB2KuxFd4WyXHkfUVxYnWrrj4xlq1QuHw84dzuGYRbvdxjvf+U684hWvwHnnnQcAmJycRDQaxcDAgOfcsbExTE5OmnNUAPE4j/nhtttuQy6XM6+1a9eucGscHBwcDh9OdHe440IELTc9tiIUCiEajXqCvR1hd3BwcOhvbNq0CT/72c/whS984bDf633vex/m5+fNa/v27Yf9ng4ODg4rhs4KvvoQfb2cS/GzXDc2xqW0220Eg0HEYjGP8HGWIAcHB4f+xS233IJ7770XDz74INasWWO+Hx8fR71ex9zcnMcatGfPHoyPj5tzfvjDH3rKY/Y4nmMjFou52DgHBweHPkVfW4IOJZmBWn10A0/uHeSsQQ4nGtycd+h3dDod3HLLLfjSl76Eb33rW9iwYYPn+EUXXYRIJIJvfvOb5runnnoK27Ztw8aNGwEAGzduxGOPPYapqSlzzv33349sNotzzjnnyDTEwcHB4QjjRHWFA/rcEhQMBtFqtQ7KHU6FkyZDON4ymLlsdw7LwfE27x1OTGzatAn33HMPvvKVryCTyZgYnlwuh0QigVwuh5tuugm33norhoaGkM1m8Y53vAMbN27EpZdeCgC48sorcc455+BNb3oTPvGJT2BychIf+MAHsGnTJmftcXBwOD7R6XRfK1FOH6KvRRBw6DFBDg4ODg7HB+644w4AwGWXXeb5/s4778Rb3vIWAMAnP/lJBINBXH/99ajVarjqqqvwmc98xpwbCoVw77334u1vfzs2btyIVCqFN7/5zfjoRz96pJrh4ODg4HAE0dciSDc+dXBwcHA4MbEcS2Y8Hsftt9+O22+/vec569evx9e+9rWVrJqDg4PDMYuVcmfrV5e4E1oE8Vr+B+pcghxOFLjFAwcHBwcHhxMcJ/hmqX0tgugGFwwuL7+DiqZOp4Nms2mu1X1QHDl0OF6hQt9eBHBwcHBwcHBwOFHQ1yJIs7o5ODg4ODg4ODg4OCwPgXb3tRLl9CP6WgTRauOsNw4ODg4ODg4ODg4HAecO178IBALL2tvHz92HVqTjGcd7+xxeOtwccXBwcHBwcDgR0fciKBAIoN1uIxQKmXTZGufjl/Sg3W6j0WggFAqZfYJY3vECR24djlcEg8ETYhHDwcHBwcHhcMJlh+tjUPAkEgkAQLPZRLPZNCKn3W77usp1Oh3E43HE43E0m01zrsuY5XC843iY361WC8Dx0RYHBwcHB4ejBrdZan8jEAggFAohGo0iGAwiGAwaQaPZ40iYmAUuFAohHA4jGo2i1Wq5DVcdHI5R0NoLdJ9ll8XRwcHBwcHB4aWir0UQY4LUnS0UCnn+AjDvKYDU6hOLxVCr1Y5mMxwcHA4CXMRwcHBwcHBwOHSc6O5wB5Vb+rbbbsPFF1+MTCaD0dFRvOENb8BTTz3lOadarWLTpk0YHh5GOp3G9ddfjz179njO2bZtG6699lokk0mMjo7i3e9+N5rN5kFXXleDaQUKBALGXQaAET2VSgWVSgWxWAyJRAKBQACNRgPZbNYQqnA47FaXHVYEnI8OLx323kaRSMT1rYODg4ODw0tFZwVffYiDYhIPPPAANm3ahIcffhj3338/Go0GrrzySpRKJXPOu971Lnz1q1/FF7/4RTzwwAPYtWsXrrvuOnO81Wrh2muvRb1ex/e//33cfffduOuuu/DBD37woCtPwcI4IL7sc+j6Fg6Hzb5CnU4HjUbDCCe3aaTDSoJzygXxrwx0wcO5wjk4ODg4ODi8VByUO9x9993n+XzXXXdhdHQUW7Zswate9SrMz8/jc5/7HO655x5cfvnlAIA777wTZ599Nh5++GFceuml+PrXv44nnngC3/jGNzA2NoYLL7wQH/vYx/De974XH/7whxGNRg+5MX5kkyvyFEDAQowBrUShUAihUOiQrFEODn7QGDQHBwcHBwcHh2MNzh3uJWB+fh4AMDQ0BADYsmULGo0GrrjiCnPOWWedhXXr1mHz5s0AgM2bN+P888/H2NiYOeeqq65CPp/H448/7nufWq2GfD7veQEw2d9CoZCv+5HGBkWjUcRiMWP5abfbaDabqFariMfjJsOcg8NKgrFnznLh4ODg4ODgcEyB2eFW4tWHOGQR1G638c53vhOveMUrcN555wEAJicnEY1GMTAw4Dl3bGwMk5OT5hwVQDzOY3647bbbkMvlzGvt2rUA4HFhC4fDRgzxmK7Gk4zyOEVTrVYzWeIcUXVYSdA908WvODg4ODg4ODgcWzhkdrZp0yb87Gc/wxe+8IWVrI8v3ve+92F+ft68tm/fDmCxy1EwGDRCiCJIz9GYIR6r1WoIBoOIRCIIh/s6WZ6BE3NHDr1cMHtt1Ovg4ODg4ODgcCyA7nAr8epHHBLrv+WWW3DvvffiwQcfxJo1a8z34+PjqNfrmJub81iD9uzZg/HxcXPOD3/4Q095zB7Hc2zEYjHEYrFF32tCA439aTQavhuk8kWRFAwGUa1WzbXRaBSNRuNQuuSYAa0OjngfHdAFUzf0dGPh4ODg4ODgcMxhpTK79SnNOShLUKfTwS233IIvfelL+Na3voUNGzZ4jl900UWIRCL45je/ab576qmnsG3bNmzcuBEAsHHjRjz22GOYmpoy59x///3IZrM455xzDqrytpsbExxwHyCCn22LEV8URul02llRHA4KnHtqYQT8LUQODg4ODg4ODg7HBg7KErRp0ybcc889+MpXvoJMJmNieHK5HBKJBHK5HG666SbceuutGBoaQjabxTve8Q5s3LgRl156KQDgyiuvxDnnnIM3velN+MQnPoHJyUl84AMfwKZNm3ytPUuh2WwaEcQkCb3iL1QAcXW+0+kgHA6jXq8jGo0imUy6lXuHg4ZmHQwEAiZlO+GEtYODg4ODg8OxhhM9O9xBiaA77rgDAHDZZZd5vr/zzjvxlre8BQDwyU9+EsFgENdffz1qtRquuuoqfOYznzHnhkIh3HvvvXj729+OjRs3IpVK4c1vfjM++tGPHlIDKILU3a0X7CxduorPVNn8rMR2OXVY7rnLLWulynM4MtAYNHuvKgcHBwcHBweHYw7tTve1EuX0IQ5KBC3HQhKPx3H77bfj9ttv73nO+vXr8bWvfe1gbu0L3YwSgEl7rSKGokaTJajrUjAYRKvVMjEc4XDYlMNzet0bWCyADka49LI6OUtU/8G2MLq02A4ODg4ODg4Oxy76Oh1ao9FAu91GJBLxiJFQKIRSqYRWq2Uyvy2V/Y2Cp1arIZlMmo1UFbZ4AmDElQbCr4RVyM5q53DsglbEVCqFZrOJRqNh4swcHBwcHBwcHI5ZnOCJEfpaBFEsBINBVCoV830oFEI4HDaCKBwOIxwOe6xGdjmdTgf1et1smhoOh1GtVgEs7CnUbDY9meWYXpsJFgCvCNL3drIGQi0HPFf/OjJ9bINzgynYHRwcHBwcHBz6AQGsUEzQSy/iqKCvRdCGDRuMhadUKnnc40qlEjqdjhEoFC9MoKAxRCSvoVAI6XQa9XodtVoNxWLRI7SazSZCoZAnvTbLt0VLp9MxiRtCoRDq9brnWKvVMhanVqvlW59AIGCu02N+7/1IuIorrRvLtr/T8zTBhF85vcr1u7/fuZrefLnlqUDsJTj82mW7HS5lpbPv3cvtUdunlh+/1Ox+9TiY9i5Vz4O1OB4oZu5Ax5dTjp67Eu3vtXChWGpuH+izX30OZo4sVS8HBwcHBweHYxN9LYL+9E//FNlsFoA/ifFLmECLTS/YqbXV9U33GLLPt1MjFwoF1Go1I3Dy+bx532w2US6XUavVUKvVUKlUzOdqtYpisWhEUrFYRK1WM8KsXC6jWq2iXq+jXq+jXC6jXq+bMukiqGA/qNiKx+OeDWSZWpyveDzumzSCMVdq/dI4LC2n1WohFot5+ot1o5tivV5Hq9VCJBIxboW6x45fIotwOIxKpeIZH4KWv0ajYVzTwuGw6Xdeb8fwcOx0nP3mhQptzjO6WVIM28k67PmhbVPxat/TT4j6ib+lBIEftL297uUHe963Wi1f101axwD49qf9fLIc3eNKBS/37tJz/MS1LV50fviJE51nfr8L9p5b7H97ftpj6uDg4ODg0BfodLqvlSinD9HXIojwywpnk+eDyfLWq4xe6bd5rZ6fyWSQTqfN56GhIc+56h5Hgk6CawsxtfRo7BFd+LSONqnX77UsCgP9rHVT6xbJpJJWJbpajgqEdrvtG4fFdlDcUbA0Gg0jBG0rWalUQrPZRKvVQqPRwNzcnBFQ7XbbCMNms2neNxoNIxZZdrPZNJ/VsmePPa14fmKJ7VYRaI8b+0vL9ptf2md+/WSfq5/9wH63j/udbwuTpcSTuv3xM+vHvzrWKuxsocA5xM869wKBACKRiKdOnKv8Ts/V9vWy5qhlVeti34f11b3G/Cx99n20/g4ODg4ODv0ClyL7OIAfweu1CmyTsaXK6LXKu9R1Sn61DHul2SbUupJtWwnse9uChOX0ch+ySbzW266H3z1s2Hsy+RFTPzGl7aSYodBhUgEKF1u00arG6yqVihEq7XbbiBqKHD2X36uI4nsdF1uY+llt+Fn72x4jP1DM2ZaMXgJJx8geF62vnzD3E8FLuZbZ4+Y3/rYFrN1umzFRCyJfFPW22yePc4xYFseHfcnxoVjmdbR08mX3l35ni0sVM2qt036wX72Evn6n93FCyMHBwcHBoT/Q1yJICZr+BeAhVH6r0X5YisTaq84HAz+CbGef62VlUqLrVw6PqfuR370Pps5LtVVJoH2On8uXX9l82VYiFT16nk389fhS1jm9BvD2kV8de4kGtQhRsGmiDbrZqXXMdvFqNBrodDrG+sCylzM2Kgg1AQdFnIoTbZOmfdf28RwVNCrqtD08zv7jNXS/pPWNnykyaXWrVCoea5yeS8FaqVSMa6ha7SiKaQVkec1m05yr7dE2BwIBRKNRT78w/b1+x7/qtqd7hdF9koIbWBBTfEUiEY8V1xZW+n6lPi/n71KCrNdCzoGuce5+Dg4ODscRjkJ2uA9/+MP4yEc+4vnuzDPPxM9//nMAQLVaxe/8zu/gC1/4gme/0bGxsRWoqBd9LYLs+ByFn4//sfQf+HLIO4Ceab2B3iLjWMDBCi9geX2y3H6z62J/Xm7f2UIsGAwiGo16jquw6VVPv3E8mP7xE55apl9ZveLfbJFrn2eXq8d5bSgUQiwW81y3XNK93GMH+/2xBIpHunZS2FHc0W2zUqksEnu0hlarVY+bqIpBFZGMJ6SAZIyhupn2qiNw4HnIthQKBaRSqSV/kxwcHBwc+geBTgeBFfg/9WDLOPfcc/GNb3zDfNb/V971rnfhn/7pn/DFL34RuVwOt9xyC6677jo89NBDL7meNvr6f7OliPZyCOZKnXMoOFz1O9wi6Fjv1+Xc66Xce7nXHgv3PJx1OpT7n0igJcaOtVOXQDszJK1PelwtV3QFtM/1+16vta3DjUbDY0nt5RKrFsI9e/bg937v94w1zcHBwcHB4VARDocxPj6+6Pv5+Xl87nOfwz333IPLL78cAHDnnXfi7LPPxsMPP4xLL710ZeuxoqU5ODg4OBhBqKtbtgXLjltcTuyWXT7L0e8PlMBFRRCTQOi96SqprrgvvvgiIpFIzyQeDg4ODg59iPb+10qUAyCfz3u+jsVii7xGAOCZZ57BxMQE4vE4Nm7ciNtuuw3r1q3Dli1b0Gg0cMUVV5hzzzrrLKxbtw6bN29ecRF08L5FDg4ODg4HhAobO9EHsJBxT/cx87u2lwVvuW6e9jnRaNQkLKHQ4f21bDsBhFqXHBwcHBz6H3SHW4kXAKxduxa5XM68brvttkX3vOSSS3DXXXfhvvvuwx133IHnn38er3zlK1EoFDA5OYloNIqBgQHPNWNjY5icnFzx9jtLkIODg8Nhhl/s1aGU4YdDjZM7GDdLTZphJ3VxcHBwcHAAgO3bt5v9OwH4WoGuueYa8/6CCy7AJZdcgvXr1+Nv//ZvkUgkjkg9CWcJcnBwcDjMsC07B5u1sdd1K1Xegc4hnBXIwcHB4ThCZwVfALLZrOflJ4JsDAwM4IwzzsCzzz6L8fFx1Ot1zM3Nec7Zs2ePbwzRS4UTQQ4ODg4ODg4ODg4nGjqdlXsdIorFIrZu3YrVq1fjoosuQiQSwTe/+U1z/KmnnsK2bduwcePGlWixB84dzsHBwcHhgGDGOgcHBwcHh0PF7/7u7+J1r3sd1q9fj127duFDH/oQQqEQfvVXfxW5XA433XQTbr31VgwNDSGbzeId73gHNm7cuOJJEQAnghwcHBwcHBwcHBxOOAQ63ddKlLNc7NixA7/6q7+K6elprFq1Cv/u3/07PPzww1i1ahUA4JOf/CSCwSCuv/56z2aphwNOBDk4ODg4HBBMj+3ighwcHByOE7xEVzZPOcvEF77whSWPx+Nx3H777bj99ttfaq0OCBcT5ODg4OCwJJjdLhwOu01xHRwcHByOCzhLkIODg4PDAeGsQA4ODg7HFwLt7mslyulHOBHk4ODg4LAkVAA5S5CDg4PDcYKj4A53LMGJIAcHBweHRbCtPs4K5ODg4OBwPMGJIAcHBwcHD5zgcXBwcDgBIBudvuRy+hBOBDk4ODg4LEKn00EgEHD7Azk4ODgcpwh0OgiswKLXSpRxNOCywzk4ODg4LAKtQZ1Ox4kgBwcHB4fjDk4EOTg4ODgsAkVQu91Gu31sp/657bbbcPHFFyOTyWB0dBRveMMb8NRTT3nOueyyyxAIBDyv3/qt3/Kcs23bNlx77bVIJpMYHR3Fu9/9bjSbzSPZFAcHB4cjByZGWIlXH8K5wzk4ODg4eKDZ4FqtFur1+lGu0dJ44IEHsGnTJlx88cVoNpt4//vfjyuvvBJPPPEEUqmUOe9tb3sbPvrRj5rPyWTSvG+1Wrj22msxPj6O73//+9i9ezd+/dd/HZFIBL//+79/RNvj4ODgcETQAbASa1z9qYGcCHJwcHBw6I1Wq4VGo3G0q7Ek7rvvPs/nu+66C6Ojo9iyZQte9apXme+TySTGx8d9y/j617+OJ554At/4xjcwNjaGCy+8EB/72Mfw3ve+Fx/+8IcRjUYPaxscHBwcHI4snDucg4ODg4MHdBcD+kME2ZifnwcADA0Neb7//Oc/j5GREZx33nl43/veh3K5bI5t3rwZ559/PsbGxsx3V111FfL5PB5//HHf+9RqNeTzec/LwcHBoV/AxAgr8epHOEuQg4ODg8MiBIPdNbJGo4FarXaUa7N8tNttvPOd78QrXvEKnHfeeeb7X/u1X8P69esxMTGBRx99FO9973vx1FNP4R/+4R8AAJOTkx4BBMB8npyc9L3Xbbfdho985COHqSUODg4OhxkdrNBmqS+9iKMBJ4IcHBwcHBaBlqBms9lXImjTpk342c9+hu9973ue72+++Wbz/vzzz8fq1avxmte8Blu3bsWpp556SPd63/veh1tvvdV8zufzWLt27aFV3MHBwcHhiMK5wzk4ODg4eKAZ1PrJHe6WW27Bvffei29/+9tYs2bNkudecsklAIBnn30WADA+Po49e/Z4zuHnXnFEsVgM2WzW83JwcHDoG5zg2eGcCHJwcHBw6Il+EEGdTge33HILvvSlL+Fb3/oWNmzYcMBrHnnkEQDA6tWrAQAbN27EY489hqmpKXPO/fffj2w2i3POOeew1NvBwcHhqKK9gq8+hHOHc3BwcHDwoCOres1m85hPkb1p0ybcc889+MpXvoJMJmNieHK5HBKJBLZu3Yp77rkHr33tazE8PIxHH30U73rXu/CqV70KF1xwAQDgyiuvxDnnnIM3velN+MQnPoHJyUl84AMfwKZNmxCLxY5m8xwcHBwcDgOcCHJwcHBw6Il+iAm64447AHQ3RFXceeedeMtb3oJoNIpvfOMb+NSnPoVSqYS1a9fi+uuvxwc+8AFzbigUwr333ou3v/3t2LhxI1KpFN785jd79hU6EDp96hLi4ODQf1iJ35uVyuzmssM5ODg4OBx3aLfbaLVaR7saS+JAZGDt2rV44IEHDljO+vXr8bWvfe2Q6zE9PX3I1zo4ODgcDAqFAnK53EsrZKXieZwIcnBwcHA4HsDMcEB/iKBjBdyXaNu2bS+dnBxjYOa77du3H3cJII7ntgHHd/tOxLZ1Oh0UCgVMTEwcxdodH3AiyMHBwcGhJxqNBqrV6tGuRl+AeyvlcrnjjpARx3MWvOO5bcDx3b4TrW0rtsjiLEEODg4ODg7+aLfbaDabR7saDg4ODg4rjRNcBLkU2Q4ODg4OPeHc4RwcHBwcjkc4S5CDg4ODQ0+0Wq1jPkX2sYJYLIYPfehDx2VKbde2/sXx3D7XtpeINoDAAc9aXjl9iECnD3N65vN55HI5zM/PH7c+oA4ODg7HAh566CHcf//9+Mu//EsAXcvQzp073e+vg4ODQ5+CPPqKM25FOPTSRVazVcM3nv5/++7/BecO5+Dg4ODQE61WC41G42hXw8HBwcHBYUXh3OEcHBwcHDxQB4Fms+nc4RwcHByOR5zgiRGcCHJwcHBwWIROp4NOp4N6vY5KpXK0q+Pg4ODgsNJod4DACgiYdn+KIOcO5+Dg4ODQE41GA7Va7WhXw8HBwcHBYUXhRJCDg4ODwyIEAgF0Oh20Wi23T9Aycfvtt+Pkk09GPB7HJZdcgh/+8IdHu0oHjQ9/+MMIBAKe11lnnWWOV6tVbNq0CcPDw0in07j++uuxZ8+eo1jj3njwwQfxute9DhMTEwgEAvjyl7/sOd7pdPDBD34Qq1evRiKRwBVXXIFnnnnGc87MzAxuvPFGZLNZDAwM4KabbkKxWDyCrfDHgdr2lre8ZdE4Xn311Z5zjsW23Xbbbbj44ouRyWQwOjqKN7zhDXjqqac85yxnDm7btg3XXnstkskkRkdH8e53v/uo/44tp22XXXbZonH7rd/6Lc85K9o2usOtxKsP4USQg4ODg4MHgcBCzlS3Wery8Dd/8ze49dZb8aEPfQg//vGP8bKXvQxXXXUVpqamjnbVDhrnnnsudu/ebV7f+973zLF3vetd+OpXv4ovfvGLeOCBB7Br1y5cd911R7G2vVEqlfCyl70Mt99+u+/xT3ziE/j0pz+Nz372s/jBD36AVCqFq666CtVq1Zxz44034vHHH8f999+Pe++9Fw8++CBuvvnmI9WEnjhQ2wDg6quv9ozjX//1X3uOH4tte+CBB7Bp0yY8/PDDuP/++9FoNHDllVeiVCqZcw40B1utFq699lrU63V8//vfx91334277roLH/zgB49GkwyW0zYAeNvb3uYZt0984hPm2Mq3baUEUH+KIJci28HBwcFhEfhfw9/93d/hi1/8Ih5++GEALkV2L1xyySW4+OKL8ad/+qcAuv20du1avOMd78Dv/d7vHeXaLR8f/vCH8eUvfxmPPPLIomPz8/NYtWoV7rnnHvzH//gfAQA///nPcfbZZ2Pz5s249NJLj3Btl49AIIAvfelLeMMb3gCgO78nJibwO7/zO/jd3/1dAN32jY2N4a677sINN9yAJ598Eueccw5+9KMf4eUvfzkA4L777sNrX/ta7NixAxMTE0erOR7YbQO6lqC5ublFFiKiX9q2d+9ejI6O4oEHHsCrXvWqZc3Bf/7nf8Yv//IvY9euXRgbGwMAfPazn8V73/te7N27F9Fo9Gg2ycBuG9C1BF144YX41Kc+5XvNSrXNpMg+5bcRDq5Aiux2Dd947tN99/+CswQ5ODg4OCwCXTFciuwDo16vY8uWLbjiiivMd8FgEFdccQU2b958FGt2aHjmmWcwMTGBU045BTfeeCO2bdsGANiyZQsajYannWeddRbWrVvXd+18/vnnMTk56WlLLpfDJZdcYtqyefNmDAwMGJEAAFdccQWCwSB+8IMfHPE6Hyy+853vYHR0FGeeeSbe/va3Y3p62hzrl7bNz88DAIaGhgAsbw5u3rwZ559/vhEJAHDVVVchn8/j8ccfP4K1Xxp224jPf/7zGBkZwXnnnYf3ve99KJfL5tiKt+0Ed4dz2eEcHBwcHHqi1Wq5FNkHwL59+9BqtTzEBADGxsbw85///CjV6tBwySWX4K677sKZZ56J3bt34yMf+Qhe+cpX4mc/+xkmJycRjUYxMDDguWZsbAyTk5NHp8KHCNbXb8x4bHJyEqOjo57j4XAYQ0NDx3x7r776alx33XXYsGEDtm7dive///245pprsHnzZoRCob5oW7vdxjvf+U684hWvwHnnnQcAy5qDk5OTvuPKY8cC/NoGAL/2a7+G9evXY2JiAo8++ije+9734qmnnsI//MM/ADgMbWuvkCtbn2aHcyLIwcHBwaEnGo2GJ0bC4fjGNddcY95fcMEFuOSSS7B+/Xr87d/+LRKJxFGsmcPB4IYbbjDvzz//fFxwwQU49dRT8Z3vfAevec1rjmLNlo9NmzbhZz/7mScm7XhBr7ZpTNb555+P1atX4zWveQ22bt2KU0899UhX87iHc4dzcHBwcOgJt1nqgTEyMoJQKLQoQ9WePXswPj5+lGq1MhgYGMAZZ5yBZ599FuPj46jX65ibm/Oc04/tZH2XGrPx8fFFiS2azSZmZmb6rr2nnHIKRkZG8OyzzwI49tt2yy23/P/t3WtMVNe7BvAHKDOCOiAizAwCRbRaCmqLQiam1BwooMZ4Sw5eYtEYPOrQ1GsNRkVtUxv9pzFao+eL2g9eWhOtqbGmKIqxIlaqQaUlQrDUymCK4SKgAvs9HzzsuAUFZZxhZp5fshNm7TV71vuyh2HNXmttnDx5EufOncPQoUPV8p6cg0ajscvfa8c+Z3tRbF1JTEwEAM3vza6xiWK/zQWxE0RERC/EOUHd0+l0iI+Px9mzZ9UyRVFw9uxZWCwWJ7as9x4+fIiKigqYTCbEx8fD19dXE2dZWRmqqqpcLs6oqCgYjUZNLA0NDSgqKlJjsVgsqKurQ3FxsVonPz8fiqKo/5y6irt376K2thYmkwlA341NRJCdnY3jx48jPz8fUVFRmv09OQctFgtu3Lih6eTl5eXBYDAgJibGMYF0obvYutKxQMmzvze7xsY5QURERF1jJ6hnVq5ciczMTIwbNw4JCQnYsWMHmpqasHDhQmc37ZWsXr0aU6dORWRkJO7du4fc3Fz4+Phgzpw5CAgIwKJFi7By5UoEBQXBYDDg008/hcVi6ZMrwz18+FD9Bh14uhjC9evXERQUhIiICCxfvhxffvklRowYgaioKGzYsAFms1ldZe3dd99Feno6srKysHfvXrS2tiI7OxuzZ892+uppL4stKCgImzdvxqxZs2A0GlFRUYHPP/8cw4cPR1paGoC+G5vVasWhQ4dw4sQJDBw4UJ3nEhAQAD8/vx6dg6mpqYiJicH8+fOxbds22Gw2rF+/HlarFXp971dCe1OxVVRU4NChQ5g8eTIGDx6MkpISrFixAklJSRg9enSfjs1liQuqr68XAFJfX+/sphARubUdO3ZITEyMhIeHS3h4uISFhfHv7wvs2rVLIiIiRKfTSUJCgly+fNnZTXplGRkZYjKZRKfTSVhYmGRkZEh5ebm6v6WlRZYtWyaDBg0Sf39/mTFjhlRXVzuxxS927ty5jlnfmi0zM1NERBRFkQ0bNkhoaKjo9XpJTk6WsrIyzTFqa2tlzpw5MmDAADEYDLJw4UJpbGx0QjRaL4utublZUlNTZciQIeLr6yuRkZGSlZUlNptNc4y+GFtXMQGQ/fv3q3V6cg7euXNHJk2aJH5+fhIcHCyrVq2S1tZWB0ej1V1sVVVVkpSUJEFBQaLX62X48OGyZs2aTn9r7RFbx//RKWFLJD38s15vKWFLXPJzgfcJIiIijWc/Fv7zn/9g9+7dUJSnY755nyAiItem3ifI/D/2u0/Qvf91uc8FzgkiIqIXUhRF7QARERG5C84JIiIiIiLyNAL7LGrgcmPKnnqlK0Fbt27F+PHjMXDgQISEhGD69OkoKyvT1Jk4caJ6p/GObcmSJZo6VVVVmDJlCvz9/RESEoI1a9agra2t99EQEREREVH3uDpczxUUFMBqtWL8+PFoa2vDunXrkJqaitLSUvTv31+tl5WVhS1btqiP/f391Z/b29sxZcoUGI1GXLp0CdXV1fjkk0/g6+uLr776yg4hERERERERvdgrdYJOnz6teXzgwAGEhISguLgYSUlJarm/v/8Lb9r0yy+/oLS0FGfOnEFoaCjGjh2LL774AmvXrsWmTZug0+leIwwiIiIiIuoxRQFghzmfLjpvtFcLI9TX1wMAgoKCNOUHDx5EcHAwYmNjkZOTg+bmZnVfYWEh4uLiEBoaqpalpaWhoaEBt27d6vJ1Hj9+jIaGBs1GRERvxrPDmQHtanFEROQmOBzu9SiKguXLl2PChAmIjY1Vy+fOnYvIyEiYzWaUlJRg7dq1KCsrw7FjxwAANptN0wECoD7uuHHU87Zu3YrNmze/blOJiOg1iYjaCeroFBEREbm61+4EWa1W3Lx5ExcvXtSUL168WP05Li4OJpMJycnJqKioQHR09Gu9Vk5ODlauXKk+bmhoQHh4+Os1nIiIekxEoCgKfHx8nN0UIiKyJ3tdxfGkK0HZ2dk4efIkLly4gKFDh760bmJiIgCgvLwc0dHRMBqNuHLliqZOTU0NALxwHpFer4de3/ubORERUfc4/I2IyAMoArusb6245mfGK80JEhFkZ2fj+PHjyM/PR1RUVLfPuX79OgDAZDIBACwWC27cuIH79++rdfLy8mAwGBATE/MqzSEiojfMy8sLPj4+HApHRERu5ZWuBFmtVhw6dAgnTpzAwIED1Tk8AQEB8PPzQ0VFBQ4dOoTJkydj8ODBKCkpwYoVK5CUlITRo0cDAFJTUxETE4P58+dj27ZtsNlsWL9+PaxWK6/2EBH1EYqiwNv76fdknBNEROR+RBSI9H5lN3scwxleqRO0Z88eAE9viPqs/fv3Y8GCBdDpdDhz5gx27NiBpqYmhIeHY9asWVi/fr1a18fHBydPnsTSpUthsVjQv39/ZGZmau4rREREzvXskDgRYQeIiMjdiNhnKJuLDqF+pU5Qd+PEw8PDUVBQ0O1xIiMjcerUqVd56S7bwaWyiYjsT1EUdTGElpYWtLe3A3h6JUj5//tBcN4QERG5stdeHc6ZGhsbAYArxBEROUljYyMCAgKc3QwiInpdYqeFEVz0SzGX7ASZzWaUlpYiJiYGf//9NwwGg7Ob1Cd0LB3OnDzFfHTGnHTGnHT2spyICBobG2E2m53UOiIisgtFAbzsMJ/HE+YE9RXe3t4ICwsDABgMBv7j8hzmRIv56Iw56Yw56exFOeEVICIicnUu2QkiIiIiIqJe4HA4IiIiIiLyJKIoEDsMh3PVJbJf6WapfYler0dubi7vLfQM5kSL+eiMOemMOemMOSEiInfnJVznlIiIiIjIIzQ0NCAgIAD/5ZeBt7x0vT5emzxBfsv3qK+vd6m5tRwOR0RERETkaRQBvDx3TpDLDocjIiIiIiJ6HbwSRERERETkaUQA2OM+Qa55JYidICIiIiIiDyOKQOwwHM5VlxdwyeFwu3fvxttvv41+/fohMTERV65ccXaTHGbTpk3w8vLSbKNGjVL3P3r0CFarFYMHD8aAAQMwa9Ys1NTUOLHF9nfhwgVMnToVZrMZXl5e+PHHHzX7RQQbN26EyWSCn58fUlJScPv2bU2dBw8eYN68eTAYDAgMDMSiRYvw8OFDB0ZhX93lZMGCBZ3Om/T0dE0dd8rJ1q1bMX78eAwcOBAhISGYPn06ysrKNHV68l6pqqrClClT4O/vj5CQEKxZswZtbW2ODMVuepKTiRMndjpPlixZoqnjTjkhIiLP5XKdoO+//x4rV65Ebm4ufv/9d4wZMwZpaWm4f/++s5vmMO+99x6qq6vV7eLFi+q+FStW4KeffsLRo0dRUFCAe/fuYebMmU5srf01NTVhzJgx2L17d5f7t23bhp07d2Lv3r0oKipC//79kZaWhkePHql15s2bh1u3biEvLw8nT57EhQsXsHjxYkeFYHfd5QQA0tPTNefN4cOHNfvdKScFBQWwWq24fPky8vLy0NraitTUVDQ1Nal1unuvtLe3Y8qUKXjy5AkuXbqE7777DgcOHMDGjRudEVKv9SQnAJCVlaU5T7Zt26buc7ecEBF5NFHst72iPnFBQ1xMQkKCWK1W9XF7e7uYzWbZunWrE1vlOLm5uTJmzJgu99XV1Ymvr68cPXpULfvjjz8EgBQWFjqohY4FQI4fP64+VhRFjEajbN++XS2rq6sTvV4vhw8fFhGR0tJSASC//fabWufnn38WLy8v+eeffxzW9jfl+ZyIiGRmZsq0adNe+Bx3z8n9+/cFgBQUFIhIz94rp06dEm9vb7HZbGqdPXv2iMFgkMePHzs2gDfg+ZyIiHz00Ufy2WefvfA57p4TIiJPUF9fLwBkotcMSfH+715vE71mCACpr6/v0esfOXJEdDqd7Nu3T27duiVZWVkSGBgoNTU1bzhyLZe6EvTkyRMUFxcjJSVFLfP29kZKSgoKCwud2DLHun37NsxmM4YNG4Z58+ahqqoKAFBcXIzW1lZNfkaNGoWIiAiPyU9lZSVsNpsmBwEBAUhMTFRzUFhYiMDAQIwbN06tk5KSAm9vbxQVFTm8zY5y/vx5hISEYOTIkVi6dClqa2vVfe6ek/r6egBAUFAQgJ69VwoLCxEXF4fQ0FC1TlpaGhoaGnDr1i0Htv7NeD4nHQ4ePIjg4GDExsYiJycHzc3N6j53zwkREb1533zzDbKysrBw4ULExMRg79698Pf3x759+xzaDpdaGOHff/9Fe3u75gMYAEJDQ/Hnn386qVWOlZiYiAMHDmDkyJGorq7G5s2b8eGHH+LmzZuw2WzQ6XQIDAzUPCc0NBQ2m805DXawjji7Okc69tlsNoSEhGj2v/XWWwgKCnLbPKWnp2PmzJmIiopCRUUF1q1bh0mTJqGwsBA+Pj5unRNFUbB8+XJMmDABsbGxANCj94rNZuvyPOrY58q6ygkAzJ07F5GRkTCbzSgpKcHatWtRVlaGY8eOAXDvnBAReRxRYJ/V4Xp+jI4LGjk5OWqZsy5ouFQniIBJkyapP48ePRqJiYmIjIzEDz/8AD8/Pye2jPqy2bNnqz/HxcVh9OjRiI6Oxvnz55GcnOzElr15VqsVN2/e1Myd83Qvysmzc8Di4uJgMpmQnJyMiooKREdHO7qZRET0BrWhFbDDwm5taAUANDQ0aMr1ej30er2mrC9d0HCpTlBwcDB8fHw6reBUU1MDo9HopFY5V2BgIN555x2Ul5fj448/xpMnT1BXV6f5htuT8tMRZ01NDUwmk1peU1ODsWPHqnWeX0ijra0NDx488Jg8DRs2DMHBwSgvL0dycrLb5iQ7O1td5GHo0KFqudFo7Pa9YjQaO03U7Pjb44456UpiYiIAoLy8HNHR0W6bEyIiT6LT6WA0GnHRdspuxxwwYADCw8M1Zbm5udi0aZPdXsPeXGpOkE6nQ3x8PM6ePauWKYqCs2fPwmKxOLFlzvPw4UNUVFTAZDIhPj4evr6+mvyUlZWhqqrKY/ITFRUFo9GoyUFDQwOKiorUHFgsFtTV1aG4uFitk5+fD0VR1H/63N3du3dRW1urdhTdLSciguzsbBw/fhz5+fmIiorS7O/Je8ViseDGjRuazmFeXh4MBgNiYmIcE4gddZeTrly/fh0ANOeJO+WEiMgT9evXD5WVlaivr7fbdvfu3U5lzw5569CnLmg4dBkGOzhy5Ijo9Xo5cOCAlJaWyuLFiyUwMFCzWpE7W7VqlZw/f14qKyvl119/lZSUFAkODpb79++LiMiSJUskIiJC8vPz5erVq2KxWMRisTi51fbV2Ngo165dk2vXrgkA+eabb+TatWvy119/iYjI119/LYGBgXLixAkpKSmRadOmSVRUlLS0tKjHSE9Pl/fff1+Kiork4sWLMmLECJkzZ46zQuq1l+WksbFRVq9eLYWFhVJZWSlnzpyRDz74QEaMGCGPHj1Sj+FOOVm6dKkEBATI+fPnpbq6Wt2am5vVOt29V9ra2iQ2NlZSU1Pl+vXrcvr0aRkyZIjk5OQ4I6Re6y4n5eXlsmXLFrl69apUVlbKiRMnZNiwYZKUlKQew91yQkREjpeQkCDZ2dnq4/b2dgkLC3P4Ss8u1wkSEdm1a5dERESITqeThIQEuXz5srOb5DAZGRliMplEp9NJWFiYZGRkSHl5ubq/paVFli1bJoMGDRJ/f3+ZMWOGVFdXO7HF9nfu3DnB01Gsmi0zM1NEni6TvWHDBgkNDRW9Xi/JyclSVlamOUZtba3MmTNHBgwYIAaDQRYuXCiNjY1OiMY+XpaT5uZmSU1NlSFDhoivr69ERkZKVlZWpy8O3CknXeUCgOzfv1+t05P3yp07d2TSpEni5+cnwcHBsmrVKmltbXVwNPbRXU6qqqokKSlJgoKCRK/Xy/Dhw2XNmjWdljx1p5wQEZHj9ZULGl4iYocpUURERERERN379ttvsX37dthsNowdOxY7d+50+PB7doKIiIiIiMijuNTCCERERERERL3FThAREREREXkUdoKIiIiIiMijsBNEREREREQehZ0gIiIiIiLyKOwEERERERGRR2EniIiIiIiIPAo7QURERERE5FHYCSIiIiIiIo/CThAREREREXkUdoKIiIiIiMijsBNEREREREQe5f8AyepOasWrYIsAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "---\n",
        "\n",
        "# **Generate Multi-Class Segmentation Masks from COCO Annotations**\n",
        "\n",
        "\n",
        "---\n",
        "\n"
      ],
      "metadata": {
        "id": "OOP5XYJVKFmF"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import json\n",
        "import os\n",
        "import numpy as np\n",
        "import cv2\n",
        "from tqdm import tqdm\n",
        "from PIL import Image\n",
        "\n",
        "# Set your paths\n",
        "coco_json_path = \"/content/drive/My Drive/CrackDetection/train/_annotations.coco.json\"  # path to annotation file\n",
        "image_folder = '/content/drive/My Drive/CrackDetection/train/images/'              # folder with original images\n",
        "mask_output_folder = '/content/drive/My Drive/CrackDetection/train/mkimages/'         # folder where masks will be saved\n",
        "\n",
        "os.makedirs(mask_output_folder, exist_ok=True)\n",
        "\n",
        "# Load COCO JSON\n",
        "with open(coco_json_path) as f:\n",
        "    coco = json.load(f)\n",
        "\n",
        "# Create a dictionary to map image_id to file_name\n",
        "image_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}\n",
        "\n",
        "# Organize annotations per image_id\n",
        "from collections import defaultdict\n",
        "image_id_to_annotations = defaultdict(list)\n",
        "for ann in coco['annotations']:\n",
        "    image_id_to_annotations[ann['image_id']].append(ann)\n",
        "\n",
        "# Generate mask for each image\n",
        "for image in tqdm(coco['images']):\n",
        "    img_id = image['id']\n",
        "    img_name = image['file_name']\n",
        "    height, width = image['height'], image['width']\n",
        "\n",
        "    # Create a blank mask\n",
        "    mask = np.zeros((height, width), dtype=np.uint8)\n",
        "\n",
        "    # Get all annotations for this image\n",
        "    annotations = image_id_to_annotations[img_id]\n",
        "\n",
        "    for ann in annotations:\n",
        "        category_id = ann['category_id']\n",
        "        segmentation = ann['segmentation']  # list of polygons\n",
        "\n",
        "        for polygon in segmentation:\n",
        "            # Convert polygon to (x, y) shape\n",
        "            pts = np.array(polygon).reshape((-1, 2)).astype(np.int32)\n",
        "            cv2.fillPoly(mask, [pts], color=category_id)\n",
        "\n",
        "    # Save mask\n",
        "    mask_path = os.path.join(mask_output_folder, os.path.splitext(img_name)[0] + \".png\")\n",
        "    Image.fromarray(mask).save(mask_path)\n",
        "os.makedirs(mask_output_folder, exist_ok=True)\n",
        "\n",
        "# Load COCO JSON\n",
        "with open(coco_json_path) as f:\n",
        "    coco = json.load(f)\n",
        "\n",
        "# Create a dictionary to map image_id to file_name\n",
        "image_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}\n",
        "\n",
        "# Organize annotations per image_id\n",
        "from collections import defaultdict\n",
        "image_id_to_annotations = defaultdict(list)\n",
        "for ann in coco['annotations']:\n",
        "    image_id_to_annotations[ann['image_id']].append(ann)\n",
        "\n",
        "# Generate mask for each image\n",
        "for image in tqdm(coco['images']):\n",
        "    img_id = image['id']\n",
        "    img_name = image['file_name']\n",
        "    height, width = image['height'], image['width']\n",
        "\n",
        "    # Create a blank mask\n",
        "    mask = np.zeros((height, width), dtype=np.uint8)\n",
        "\n",
        "    # Get all annotations for this image\n",
        "    annotations = image_id_to_annotations[img_id]\n",
        "\n",
        "    for ann in annotations:\n",
        "        category_id = ann['category_id']\n",
        "        segmentation = ann['segmentation']  # list of polygons\n",
        "\n",
        "        for polygon in segmentation:\n",
        "            # Convert polygon to (x, y) shape\n",
        "            pts = np.array(polygon).reshape((-1, 2)).astype(np.int32)\n",
        "            cv2.fillPoly(mask, [pts], color=category_id)\n",
        "\n",
        "    # Save mask\n",
        "    mask_path = os.path.join(mask_output_folder, os.path.splitext(img_name)[0] + \".png\")\n",
        "    Image.fromarray(mask).save(mask_path)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vQIn-CVE_zLg",
        "outputId": "aaf66697-1bc7-448b-cc99-30d41b19b8a1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 753/753 [00:17<00:00, 43.14it/s]\n",
            "100%|██████████| 753/753 [00:17<00:00, 43.73it/s]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "***Display Crack Type Labels and Class IDs***"
      ],
      "metadata": {
        "id": "pnPn8CsZKWvk"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "for cat in coco['categories']:\n",
        "    print(f\"Class ID {cat['id']} = {cat['name']}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uR627WIc_5hK",
        "outputId": "face9d31-cbc6-409f-b471-434c0950b882"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Class ID 0 = Cracks\n",
            "Class ID 1 = Compression Crack\n",
            "Class ID 2 = Rebar Detachment\n",
            "Class ID 3 = Shear Type - 01\n",
            "Class ID 4 = Shear Type - 02\n",
            "Class ID 5 = Tension Crack\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "---\n",
        "\n",
        "# **Create a Custom Dataset Class for Crack Segmentation**\n",
        "---\n",
        "\n"
      ],
      "metadata": {
        "id": "ua_SZzMJKrTX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install albumentations\n",
        "import torch\n",
        "from torch.utils.data import Dataset\n",
        "import cv2\n",
        "import os\n",
        "import numpy as np\n",
        "import albumentations as A\n",
        "from albumentations.pytorch import ToTensorV2\n",
        "\n",
        "class CrackSegmentationDataset(Dataset):\n",
        "    def __init__(self, image_dir, mask_dir, transform=None):\n",
        "        self.image_dir = image_dir\n",
        "        self.mask_dir = mask_dir\n",
        "        self.image_names = sorted(os.listdir(image_dir))\n",
        "        self.mask_names = sorted(os.listdir(mask_dir))\n",
        "        self.transform = transform\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.image_names)\n",
        "\n",
        "    def __getitem__(self, idx):\n",
        "        # Load image and mask\n",
        "        img_path = os.path.join(self.image_dir, self.image_names[idx])\n",
        "        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])\n",
        "\n",
        "        image = cv2.imread(img_path)\n",
        "        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\n",
        "        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)\n",
        "\n",
        "        # Apply transformations\n",
        "        if self.transform:\n",
        "            augmented = self.transform(image=image, mask=mask)\n",
        "            image = augmented['image']\n",
        "            mask = augmented['mask']\n",
        "\n",
        "        return image, mask.long()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "j9ruEK4w_7f_",
        "outputId": "cf52f1ff-48c5-4a89-d9c2-94047c0a9adb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: albumentations in /usr/local/lib/python3.11/dist-packages (2.0.5)\n",
            "Requirement already satisfied: numpy>=1.24.4 in /usr/local/lib/python3.11/dist-packages (from albumentations) (2.0.2)\n",
            "Requirement already satisfied: scipy>=1.10.0 in /usr/local/lib/python3.11/dist-packages (from albumentations) (1.14.1)\n",
            "Requirement already satisfied: PyYAML in /usr/local/lib/python3.11/dist-packages (from albumentations) (6.0.2)\n",
            "Requirement already satisfied: pydantic>=2.9.2 in /usr/local/lib/python3.11/dist-packages (from albumentations) (2.11.3)\n",
            "Requirement already satisfied: albucore==0.0.23 in /usr/local/lib/python3.11/dist-packages (from albumentations) (0.0.23)\n",
            "Requirement already satisfied: opencv-python-headless>=4.9.0.80 in /usr/local/lib/python3.11/dist-packages (from albumentations) (4.10.0.84)\n",
            "Requirement already satisfied: stringzilla>=3.10.4 in /usr/local/lib/python3.11/dist-packages (from albucore==0.0.23->albumentations) (3.12.4)\n",
            "Requirement already satisfied: simsimd>=5.9.2 in /usr/local/lib/python3.11/dist-packages (from albucore==0.0.23->albumentations) (6.2.1)\n",
            "Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.11/dist-packages (from pydantic>=2.9.2->albumentations) (0.7.0)\n",
            "Requirement already satisfied: pydantic-core==2.33.1 in /usr/local/lib/python3.11/dist-packages (from pydantic>=2.9.2->albumentations) (2.33.1)\n",
            "Requirement already satisfied: typing-extensions>=4.12.2 in /usr/local/lib/python3.11/dist-packages (from pydantic>=2.9.2->albumentations) (4.13.2)\n",
            "Requirement already satisfied: typing-inspection>=0.4.0 in /usr/local/lib/python3.11/dist-packages (from pydantic>=2.9.2->albumentations) (0.4.0)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "---\n",
        "\n",
        "# **Define Transformations and Create DataLoader**\n",
        "\n",
        "---\n",
        "\n"
      ],
      "metadata": {
        "id": "aSzvTM7-K5RN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "transform = A.Compose([\n",
        "    A.Resize(256, 256),\n",
        "    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),\n",
        "    ToTensorV2()\n",
        "\n",
        "])\n",
        "train_dataset = CrackSegmentationDataset(\n",
        "    image_dir='/content/drive/My Drive/CrackDetection/train/images',\n",
        "    mask_dir='/content/drive/My Drive/CrackDetection/train/mkimages',\n",
        "    transform=transform\n",
        ")\n",
        "\n",
        "train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)\n"
      ],
      "metadata": {
        "id": "t1oOqzVo_9TX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "---\n",
        "\n",
        "#**Visualize Image and Corresponding Segmentation Mask**\n",
        "---\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "hUh15RSbLQe-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Helper to plot image and mask\n",
        "def visualize_sample(img_tensor, mask_tensor):\n",
        "    image = img_tensor.permute(1, 2, 0).numpy()  # convert from CxHxW to HxWxC\n",
        "    mask = mask_tensor.numpy()\n",
        "\n",
        "    plt.figure(figsize=(10, 4))\n",
        "    plt.subplot(1, 2, 1)\n",
        "    plt.imshow(image)\n",
        "    plt.title(\"Image\")\n",
        "    plt.axis('off')\n",
        "\n",
        "    plt.subplot(1, 2, 2)\n",
        "    plt.imshow(mask, cmap='jet', vmin=0, vmax=5)  # 0 to 5 for your 6 classes\n",
        "    plt.title(\"Segmentation Mask\")\n",
        "    plt.axis('off')\n",
        "\n",
        "    plt.show()\n",
        "\n",
        "# Get a batch from DataLoader\n",
        "for imgs, masks in train_loader:\n",
        "    for i in range(2):  # visualize 2 samples from the batch\n",
        "        visualize_sample(imgs[i], masks[i])\n",
        "    break\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 717
        },
        "id": "xniVCDi4__a5",
        "outputId": "aab121fa-7957-40c2-fb86-20d3c695673e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x400 with 2 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAu4AAAFeCAYAAADaP5oiAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAA8PhJREFUeJzsvXmYZVV5Nb7uPNbQ8wjNDAKOiH4OCOKAiKISxUAUxaj5EiHGxBjzJSqawST+kjgkn5ro5xAhKk4xKnHEIQqKiggiytg0dNP0VNOdh/P7o7J2r/PWPlW3eqqq7r2e5z517xn2fOqs991rvzsVRVGEgICAgICAgICAgIBFjfRCFyAgICAgICAgICAgYG4E4h4QEBAQEBAQEBCwBBCIe0BAQEBAQEBAQMASQCDuAQEBAQEBAQEBAUsAgbgHBAQEBAQEBAQELAEE4h4QEBAQEBAQEBCwBBCIe0BAQEBAQEBAQMASQCDuAQEBAQEBAQEBAUsAgbgHBAQEBAQEBAQELAEE4h4QEBAQEBAQcACQSqVw1VVXLXQxFgQf/ehHkUql8OMf/3ihi3JYIxD3gHkhPJgBAQEBSwu33norXvziF2PTpk0oFovYsGEDnvWsZ+F973vfQhftkGPr1q246qqr8LOf/Wyf0/jKV76y6Mj5VVddhVQqhXQ6jS1btsw4PzExgVKphFQqhSuuuGIBShhwoBCIe0BAQEBAwGGKH/zgB3j84x+PW265Ba95zWvwT//0T3j1q1+NdDqN97znPQtdvEOOrVu34u1vf/t+E/e3v/3t3nONRgN//ud/vs9p7y8KhQL+/d//fcbxz33ucwtQmoCDgexCFyAgICAgICDg4OCv/uqvMDIygptuugmjo6Oxcw8//PDCFOowRrFYXND8n/vc5+Lf//3f8aY3vSl2/JprrsEFF1yAz372swtUsoADheBxD9gvvPKVr0S1WsX999+P5z3veahWq9iwYQP++Z//GcD0FO25556LSqWCTZs24Zprrondv3v3brzxjW/EIx/5SFSrVQwPD+P888/HLbfcMiOvzZs348ILL0SlUsHq1avxhje8AV/96leRSqXw7W9/O3btD3/4QzznOc/ByMgIyuUyzj77bHz/+98/aO0QEBAQsBhx991347TTTptB2gFg9erVM4594hOfwBlnnIFSqYTly5fjN3/zN73Si3/+53/Gcccdh1KphCc84Qn43ve+h3POOQfnnHOOu+bb3/42UqkUPv3pT+Ptb387NmzYgKGhIbz4xS/G+Pg4Wq0W/uAP/gCrV69GtVrF5ZdfjlartU9lOuecc3D66afj9ttvx9Of/nSUy2Vs2LABf/d3fxcrz5lnngkAuPzyy5FKpZBKpfDRj34UAPC9730PL3nJS3D00UejUCjgqKOOwhve8AY0Gg2Xxitf+Ur3fuP9qVTKnfdp3G+++Wacf/75GB4eRrVaxTOe8QzceOONsWsoQ/3+97+PP/zDP8SqVatQqVTwohe9CDt27JjRJkm49NJL8bOf/Qx33HGHO/bQQw/hW9/6Fi699NIZ17fbbbz1rW/FGWecgZGREVQqFZx11lm4/vrrZ1z7yU9+EmeccQaGhoYwPDyMRz7ykXPO2uzZswdPeMITsHHjRvzqV78auB4ByQge94D9Rq/Xw/nnn4+nPe1p+Lu/+ztcffXVuOKKK1CpVPBnf/Zn+K3f+i1cdNFF+MAHPoDLLrsMT3rSk3DssccCAO655x584QtfwEte8hIce+yx2L59Oz74wQ/i7LPPxu23347169cDAGq1Gs4991xs27YNr3/967F27Vpcc8013n8u3/rWt3D++efjjDPOwNve9jak02l85CMfwbnnnovvfe97eMITnnBI2ycgICBgobBp0ybccMMNuO2223D66afPeu1f/dVf4S1veQsuvvhivPrVr8aOHTvwvve9D0972tNw8803O/L//ve/H1dccQXOOussvOENb8B9992HF77whVi2bBk2btw4I913vvOdKJVKePOb34y77roL73vf+5DL5ZBOp7Fnzx5cddVVuPHGG/HRj34Uxx57LN761rfOu0zANEl8znOeg4suuggXX3wxPvOZz+BP/uRP8MhHPhLnn38+HvGIR+Ad73gH3vrWt+K1r30tzjrrLADAk5/8ZADAtddei3q9jt/93d/FihUr8KMf/Qjve9/78MADD+Daa68FAPzO7/wOtm7diq9//ev4t3/7tznb/xe/+AXOOussDA8P401vehNyuRw++MEP4pxzzsF3vvMdPPGJT4xdf+WVV2LZsmV429vehvvuuw/vfve7ccUVV+BTn/rUnHkBwNOe9jRs3LgR11xzDd7xjncAAD71qU+hWq3iggsumHH9xMQEPvShD+GSSy7Ba17zGkxOTuLDH/4wzjvvPPzoRz/CYx7zGADA17/+dVxyySV4xjOegb/9278FAPzyl7/E97//fbz+9a/3lmXnzp141rOehd27d+M73/kOjj/++IHqEDAHooCAeeAjH/lIBCC66aaboiiKole84hURgOiv//qv3TV79uyJSqVSlEqlok9+8pPu+B133BEBiN72tre5Y81mM+r1erE87r333qhQKETveMc73LG///u/jwBEX/jCF9yxRqMRnXLKKRGA6Prrr4+iKIr6/X504oknRuedd17U7/fdtfV6PTr22GOjZz3rWQekHQICAgKWAr72ta9FmUwmymQy0ZOe9KToTW96U/TVr341arfbsevuu+++KJPJRH/1V38VO37rrbdG2WzWHW+1WtGKFSuiM888M+p0Ou66j370oxGA6Oyzz3bHrr/++ghAdPrpp8fyu+SSS6JUKhWdf/75sbye9KQnRZs2bZp3maIois4+++wIQPTxj3/cHWu1WtHatWuj3/iN33DHbrrppghA9JGPfGRGW9Xr9RnH3vnOd0apVCravHmzO/a6170uSqJP9h33whe+MMrn89Hdd9/tjm3dujUaGhqKnva0p7ljfLc+85nPjL273vCGN0SZTCYaGxvz5ke87W1viwBEO3bsiN74xjdGJ5xwgjt35plnRpdffrkr3+te9zp3rtvtRq1WK5bWnj17ojVr1kSvetWr3LHXv/710fDwcNTtdhPLoPxg27Zt0WmnnRYdd9xx0X333Tdr2QPmhyCVCTggePWrX+2+j46O4uSTT0alUsHFF1/sjp988skYHR3FPffc444VCgWk09PDsNfrYdeuXahWqzj55JPx05/+1F33X//1X9iwYQMuvPBCd6xYLOI1r3lNrBw/+9nPcOedd+LSSy/Frl27sHPnTuzcuRO1Wg3PeMYz8N3vfhf9fv+A1z8gICBgMeJZz3oWbrjhBlx44YW45ZZb8Hd/93c477zzsGHDBnzxi190133uc59Dv9/HxRdf7P5v7ty5E2vXrsWJJ57oZjd//OMfY9euXXjNa16DbHbvpP1v/dZvYdmyZd4yXHbZZcjlcu73E5/4RERRhFe96lWx6574xCdiy5Yt6Ha78yoTUa1W8bKXvcz9zufzeMITnhB758yGUqnkvtdqNezcuRNPfvKTEUURbr755oHSUPR6PXzta1/DC1/4Qhx33HHu+Lp163DppZfiv//7vzExMRG757WvfW1MenPWWWeh1+th8+bNA+d76aWX4q677sJNN93k/vpkMgCQyWSQz+cBAP1+H7t370a328XjH//42Dt4dHQUtVoNX//61+fM/4EHHsDZZ5+NTqeD7373u9i0adPAZQ+YG0EqE7DfKBaLWLVqVezYyMgINm7cGPsHxON79uxxv/v9Pt7znvfg//7f/4t7770XvV7PnVuxYoX7vnnzZhx//PEz0jvhhBNiv++8804AwCte8YrE8o6Pjye+YAICAgION5x55pn43Oc+h3a7jVtuuQWf//zn8Y//+I948YtfjJ/97Gc49dRTceeddyKKIpx44oneNEi8SSDt/95sNotjjjnGe+/RRx8d+z0yMgIAOOqoo2Yc7/f7GB8fx4oVKwYuE+F75yxbtgw///nPvfdb3H///XjrW9+KL37xi7H3FDD93pgvduzYgXq9jpNPPnnGuUc84hHo9/vYsmULTjvtNHfcthXfVbY8s+Gxj30sTjnlFFxzzTUYHR3F2rVrce655yZe/7GPfQx///d/jzvuuAOdTscdp6QVAH7v934Pn/70p3H++edjw4YNePazn42LL74Yz3nOc2ak9/KXvxzZbBa//OUvsXbt2oHLHTAYAnEP2G9kMpl5HY+iyH3/67/+a7zlLW/Bq171KvzFX/wFli9fjnQ6jT/4gz/YJ88473nXu97ltHkW1Wp13ukGBAQELHXk83mceeaZOPPMM3HSSSfh8ssvx7XXXou3ve1t6Pf7SKVSuO6667z/u/fn/+a+viPmW6ZB3jlJ6PV6To/9J3/yJzjllFNQqVTw4IMP4pWvfOUhm6ndnzooLr30Urz//e/H0NAQXvrSl7qZbYtPfOITeOUrX4kXvvCF+OM//mOsXr0amUwG73znO3H33Xe761avXo2f/exn+OpXv4rrrrsO1113HT7ykY/gsssuw8c+9rFYmhdddBE+/vGP4z3veQ/e+c53zqvcAXMjEPeABcVnPvMZPP3pT8eHP/zh2PGxsTGsXLnS/d60aRNuv/12RFEU86jcddddsfu4+GV4eBjPfOYzD2LJAwICApYuHv/4xwMAtm3bBmD6f2cURTj22GNx0kknJd5H2cNdd92Fpz/96e54t9vFfffdh0c96lEHrIyDlmk+sB554tZbb8Wvf/1rfOxjH8Nll13mjvukIUlpWKxatQrlctkbTeWOO+5AOp2eMetwoHDppZfirW99K7Zt2zbrItrPfOYzOO644/C5z30uVq+3ve1tM67N5/N4/vOfj+c///no9/v4vd/7PXzwgx/EW97yltgMzJVXXokTTjgBb33rWzEyMoI3v/nNB7ZyRziCxj1gQZHJZGZ4Eq699lo8+OCDsWPnnXceHnzwwZgms9ls4l//9V9j151xxhk4/vjj8f/9f/8fpqamZuQ3n7BaAQEBAUsd119/vddb+5WvfAUAnIzjoosuQiaTwdvf/vYZ10dRhF27dgGYJvwrVqzAv/7rvzotOgBcffXV85JzDIJByzQfVCoVANPOIQU93ZpPFEXecIdJaVhkMhk8+9nPxn/8x3/gvvvuc8e3b9+Oa665Bk996lMxPDw87zoMguOPPx7vfve78c53vnPWSGq+ev/whz/EDTfcELvOtnU6nXZGmi+E51ve8ha88Y1vxJ/+6Z/i/e9//z7XI2Amgsc9YEHxvOc9D+94xztw+eWX48lPfjJuvfVWXH311bGFPMB0CK5/+qd/wiWXXILXv/71WLduHa6++mq32QU9Bel0Gh/60Idw/vnn47TTTsPll1+ODRs24MEHH8T111+P4eFh/Od//uchr2dAQEDAQuDKK69EvV7Hi170Ipxyyilot9v4wQ9+gE996lM45phjcPnllwOYJnp/+Zd/iT/90z914R2HhoZw77334vOf/zxe+9rX4o1vfCPy+TyuuuoqXHnllTj33HNx8cUX47777sNHP/pR7zqk/cGgZZpvmqOjo/jABz6AoaEhVCoVPPGJT8Qpp5yC448/Hm984xvx4IMPYnh4GJ/97Ge9xsgZZ5wBAPj93/99nHfeechkMvjN3/xNb35/+Zd/ia9//et46lOfit/7vd9DNpvFBz/4QbRarViM+YOBpDCNiuc973n43Oc+hxe96EW44IILcO+99+IDH/gATj311Jjz69WvfjV2796Nc889Fxs3bsTmzZvxvve9D495zGPwiEc8wpv2u971LoyPj+N1r3sdhoaGYguHA/YDhzaITcBShy8cZKVSmXHd2WefHZ122mkzjm/atCm64IIL3O9msxn90R/9UbRu3bqoVCpFT3nKU6IbbrghOvvss2NhxaIoiu65557oggsuiEqlUrRq1aroj/7oj6LPfvazEYDoxhtvjF178803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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x400 with 2 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAu4AAAFeCAYAAADaP5oiAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQABAABJREFUeJzs/Xm8pVV5JY6vM8/DHWuuYhKQwSEItgoyiUg0tBJFxQTFn8Zuh6gd2yTd7ZioiXYSh6TVxLTGFhxwiEm+EFHjPKBRIAiiIFRRc935zPP7++O69l3vvu8595yqgqoLe30+93PPPed99/zes55nr+fZIc/zPDg4ODg4ODg4ODg4HNcIH+sGODg4ODg4ODg4ODisDUfcHRwcHBwcHBwcHNYBHHF3cHBwcHBwcHBwWAdwxN3BwcHBwcHBwcFhHcARdwcHBwcHBwcHB4d1AEfcHRwcHBwcHBwcHNYBHHF3cHBwcHBwcHBwWAdwxN3BwcHBwcHBwcFhHcARdwcHBwcHBwcHB4d1AEfcHRwcHBwcHByOAkKhEN7+9rcf62YcE3ziE59AKBTCv//7vx/rpjyi4Yi7w0hwD6aDg4PD+sKdd96J5z//+dixYweSySS2bNmCyy67DB/60IeOddMeduzbtw9vf/vbcfvttx92GTfddNNxR87f/va3IxQKIRwOY/fu3as+L5VKSKVSCIVCeO1rX3sMWuhwtOCIu4ODg4ODwyMU3//+9/GkJz0Jd9xxB175ylfir//6r/GKV7wC4XAYH/jAB4518x527Nu3D+94xzuOmLi/4x3vCPysXq/jf/2v/3XYZR8pEokEPv3pT696/4tf/OIxaI3DQ4HosW6Ag4ODg4ODw0ODd73rXSgUCvjxj3+MYrHo++zQoUPHplGPYCSTyWNa/2/+5m/i05/+NN785jf73r/hhhvw7Gc/G1/4wheOUcscjhacx93hiPCyl70M2WwWDz74IJ7znOcgm81iy5Yt+Ju/+RsAy1u0l1xyCTKZDHbs2IEbbrjBd//8/Dze9KY34eyzz0Y2m0U+n8cVV1yBO+64Y1Vdu3btwpVXXolMJoPp6Wm88Y1vxFe+8hWEQiF885vf9F1766234lnPehYKhQLS6TQuvPBCfO9733vIxsHBwcHheMSvfvUrnHnmmatIOwBMT0+veu9Tn/oUzjnnHKRSKYyPj+NFL3pRoPTib/7mb3DSSSchlUrhvPPOw3e+8x1cdNFFuOiii8w13/zmNxEKhfC5z30O73jHO7Blyxbkcjk8//nPx9LSEprNJt7whjdgenoa2WwW1113HZrN5mG16aKLLsJZZ52Fu+++GxdffDHS6TS2bNmC9773vb72nHvuuQCA6667DqFQCKFQCJ/4xCcAAN/5znfwghe8ANu3b0cikcC2bdvwxje+EfV63ZTxspe9zHy/8f5QKGQ+D9K433bbbbjiiiuQz+eRzWZx6aWX4oc//KHvGspQv/e97+G//bf/hqmpKWQyGTzvec/DzMzMqjHph2uuuQa333477rnnHvPegQMH8G//9m+45pprVl3farXw1re+Feeccw4KhQIymQwuuOACfOMb31h17Wc+8xmcc845yOVyyOfzOPvss9fctVlYWMB5552HrVu34he/+MXQ/XDoD+dxdzhidLtdXHHFFXj605+O9773vbj++uvx2te+FplMBv/zf/5PvOQlL8FVV12Fj3zkI7j22mvxlKc8BSeeeCIA4P7778c//uM/4gUveAFOPPFEHDx4EB/96Edx4YUX4u6778bmzZsBANVqFZdccgn279+P17/+9di4cSNuuOGGwH8u//Zv/4YrrrgC55xzDt72trchHA7j4x//OC655BJ85zvfwXnnnfewjo+Dg4PDscKOHTvwgx/8AD/72c9w1llnDbz2Xe96F97ylrfg6quvxite8QrMzMzgQx/6EJ7+9KfjtttuM+T/wx/+MF772tfiggsuwBvf+Ebs3LkTz33uczE2NoatW7euKvc973kPUqkU/uiP/gj33XcfPvShDyEWiyEcDmNhYQFvf/vb8cMf/hCf+MQncOKJJ+Ktb33ryG0Clknis571LFx11VW4+uqr8fnPfx5/+Id/iLPPPhtXXHEFHvvYx+Kd73wn3vrWt+L3fu/3cMEFFwAAnvrUpwIAbrzxRtRqNfzX//pfMTExgR/96Ef40Ic+hD179uDGG28EALzqVa/Cvn378NWvfhX/7//9vzXH/6677sIFF1yAfD6PN7/5zYjFYvjoRz+Kiy66CN/61rfw5Cc/2Xf96173OoyNjeFtb3sbdu7cife///147Wtfi89+9rNr1gUAT3/607F161bccMMNeOc73wkA+OxnP4tsNotnP/vZq64vlUr42Mc+hhe/+MV45StfiXK5jL//+7/H5Zdfjh/96Ed4whOeAAD46le/ihe/+MW49NJL8ed//ucAgJ///Of43ve+h9e//vWBbZmdncVll12G+fl5fOtb38LJJ588VB8c1oDn4DACPv7xj3sAvB//+Mee53neS1/6Ug+A9+53v9tcs7Cw4KVSKS8UCnmf+cxnzPv33HOPB8B729veZt5rNBpet9v11fHAAw94iUTCe+c732ne+4u/+AsPgPeP//iP5r16ve6dfvrpHgDvG9/4hud5ntfr9bzHPOYx3uWXX+71ej1zba1W80488UTvsssuOyrj4ODg4LAecMstt3iRSMSLRCLeU57yFO/Nb36z95WvfMVrtVq+63bu3OlFIhHvXe96l+/9O++804tGo+b9ZrPpTUxMeOeee67XbrfNdZ/4xCc8AN6FF15o3vvGN77hAfDOOussX30vfvGLvVAo5F1xxRW+up7ylKd4O3bsGLlNnud5F154oQfA++QnP2neazab3saNG73f/u3fNu/9+Mc/9gB4H//4x1eNVa1WW/Xee97zHi8UCnm7du0y773mNa/x+tEn+zvuuc99rhePx71f/epX5r19+/Z5uVzOe/rTn27e43frM57xDN931xvf+EYvEol4i4uLgfURb3vb2zwA3szMjPemN73JO+WUU8xn5557rnfdddeZ9r3mNa8xn3U6Ha/ZbPrKWlhY8DZs2OC9/OUvN++9/vWv9/L5vNfpdPq2QfnB/v37vTPPPNM76aSTvJ07dw5su8NocFIZh6OCV7ziFeZ1sVjEaaedhkwmg6uvvtq8f9ppp6FYLOL+++837yUSCYTDy8uw2+1ibm4O2WwWp512Gn7605+a6/71X/8VW7ZswZVXXmneSyaTeOUrX+lrx+233457770X11xzDebm5jA7O4vZ2VlUq1Vceuml+Pa3v41er3fU++/g4OBwPOKyyy7DD37wA1x55ZW444478N73vheXX345tmzZgn/6p38y133xi19Er9fD1Vdfbf5vzs7OYuPGjXjMYx5jdjf//d//HXNzc3jlK1+JaHRl0/4lL3kJxsbGAttw7bXXIhaLmb+f/OQnw/M8vPzlL/dd9+QnPxm7d+9Gp9MZqU1ENpvF7/zO75i/4/E4zjvvPN93ziCkUinzulqtYnZ2Fk996lPheR5uu+22ocpQdLtd3HLLLXjuc5+Lk046yby/adMmXHPNNfjud7+LUqnku+f3fu/3fNKbCy64AN1uF7t27Rq63muuuQb33XcffvzjH5vfQTIZAIhEIojH4wCAXq+H+fl5dDodPOlJT/J9BxeLRVSrVXz1q19ds/49e/bgwgsvRLvdxre//W3s2LFj6LY7rA0nlXE4YiSTSUxNTfneKxQK2Lp1q+8fEN9fWFgwf/d6PXzgAx/A//k//wcPPPAAut2u+WxiYsK83rVrF04++eRV5Z1yyim+v++9914AwEtf+tK+7V1aWur7BePg4ODwSMO5556LL37xi2i1WrjjjjvwpS99CX/1V3+F5z//+bj99ttxxhln4N5774XneXjMYx4TWAaJNwmk/b83Go3ihBNOCLx3+/btvr8LhQIAYNu2bave7/V6WFpawsTExNBtIoK+c8bGxvAf//EfgffbePDBB/HWt74V//RP/+T7ngKWvzdGxczMDGq1Gk477bRVnz32sY9Fr9fD7t27ceaZZ5r37bHid5XdnkF44hOfiNNPPx033HADisUiNm7ciEsuuaTv9f/wD/+Av/iLv8A999yDdrtt3qekFQBe/epX43Of+xyuuOIKbNmyBc985jNx9dVX41nPetaq8n73d38X0WgUP//5z7Fx48ah2+0wHBxxdzhiRCKRkd73PM+8fve73423vOUtePnLX44/+ZM/wfj4OMLhMN7whjcclmec97zvfe8z2jwb2Wx25HIdHBwc1jvi8TjOPfdcnHvuuTj11FNx3XXX4cYbb8Tb3vY29Ho9hEIh3HzzzYH/u4/k/+bhfkeM2qZhvnP6odvtGj32H/7hH+L0009HJpPB3r178bKXvexh26k9kj4orrnmGnz4wx9GLpfDC1/4QrOzbeNTn/oUXvayl+G5z30u/vt//++Ynp5GJBLBe97zHvzqV78y101PT+P222/HV77yFdx88824+eab8fGPfxzXXnst/uEf/sFX5lVXXYVPfvKT+MAHPoD3vOc9I7XbYW044u5wTPH5z38eF198Mf7+7//e9/7i4iImJyfN3zt27MDdd98Nz/N8HpX77rvPdx+DX/L5PJ7xjGc8hC13cHBwWL940pOeBADYv38/gOX/nZ7n4cQTT8Spp57a9z7KHu677z5cfPHF5v1Op4OdO3ficY973FFr47BtGgW2R56488478ctf/hL/8A//gGuvvda8HyQN6VeGjampKaTT6cBsKvfccw/C4fCqXYejhWuuuQZvfetbsX///oFBtJ///Odx0kkn4Ytf/KKvX29729tWXRuPx/Fbv/Vb+K3f+i30ej28+tWvxkc/+lG85S1v8e3AvO51r8Mpp5yCt771rSgUCvijP/qjo9u5Rzmcxt3hmCISiazyJNx4443Yu3ev773LL78ce/fu9WkyG40G/u7v/s533TnnnIOTTz4Z//t//29UKpVV9Y2SVsvBwcFhveMb3/hGoLf2pptuAgAj47jqqqsQiUTwjne8Y9X1nudhbm4OwDLhn5iYwN/93d8ZLToAXH/99SPJOYbBsG0aBZlMBsCyc0hBT7fW43leYLrDfmXYiEQieOYzn4kvf/nL2Llzp3n/4MGDuOGGG3D++ecjn8+P3IdhcPLJJ+P9738/3vOe9wzMpBbU71tvvRU/+MEPfNfZYx0Oh42RFpTC8y1veQve9KY34Y//+I/x4Q9/+LD74bAazuPucEzxnOc8B+985ztx3XXX4alPfSruvPNOXH/99b5AHmA5Bddf//Vf48UvfjFe//rXY9OmTbj++uvNYRf0FITDYXzsYx/DFVdcgTPPPBPXXXcdtmzZgr179+Ib3/gG8vk8/vmf//lh76eDg4PDscDrXvc61Go1PO95z8Ppp5+OVquF73//+/jsZz+LE044Addddx2AZaL3p3/6p/jjP/5jk94xl8vhgQcewJe+9CX83u/9Ht70pjchHo/j7W9/O173utfhkksuwdVXX42dO3fiE5/4RGAc0pFg2DaNWmaxWMRHPvIR5HI5ZDIZPPnJT8bpp5+Ok08+GW9605uwd+9e5PN5fOELXwg0Rs455xwAwO///u/j8ssvRyQSwYte9KLA+v70T/8UX/3qV3H++efj1a9+NaLRKD760Y+i2Wz6csw/FOiXplHxnOc8B1/84hfxvOc9D89+9rPxwAMP4CMf+QjOOOMMn/PrFa94Bebn53HJJZdg69at2LVrFz70oQ/hCU94Ah772McGlv2+970PS0tLeM1rXoNcLucLHHY4Ajy8SWwc1juC0kFmMplV11144YXemWeeuer9HTt2eM9+9rPN341Gw/uDP/gDb9OmTV4qlfKe9rSneT/4wQ+8Cy+80JdWzPM87/777/ee/exne6lUypuamvL+4A/+wPvCF77gAfB++MMf+q697bbbvKuuusqbmJjw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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install segmentation-models-pytorch\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GbwABcEuABTE",
        "outputId": "252ff4e8-ceb9-4fec-a084-b49bb3db9cf8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: segmentation-models-pytorch in /usr/local/lib/python3.11/dist-packages (0.5.0)\n",
            "Requirement already satisfied: huggingface-hub>=0.24 in /usr/local/lib/python3.11/dist-packages (from segmentation-models-pytorch) (0.30.2)\n",
            "Requirement already satisfied: numpy>=1.19.3 in /usr/local/lib/python3.11/dist-packages (from segmentation-models-pytorch) (2.0.2)\n",
            "Requirement already satisfied: pillow>=8 in /usr/local/lib/python3.11/dist-packages (from segmentation-models-pytorch) (11.1.0)\n",
            "Requirement already satisfied: safetensors>=0.3.1 in /usr/local/lib/python3.11/dist-packages (from segmentation-models-pytorch) (0.5.3)\n",
            "Requirement already satisfied: timm>=0.9 in /usr/local/lib/python3.11/dist-packages (from segmentation-models-pytorch) (1.0.15)\n",
            "Requirement already satisfied: torch>=1.8 in /usr/local/lib/python3.11/dist-packages (from segmentation-models-pytorch) (2.6.0+cu124)\n",
            "Requirement already satisfied: torchvision>=0.9 in /usr/local/lib/python3.11/dist-packages (from segmentation-models-pytorch) (0.21.0+cu124)\n",
            "Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.11/dist-packages (from segmentation-models-pytorch) (4.67.1)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from huggingface-hub>=0.24->segmentation-models-pytorch) (3.18.0)\n",
            "Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub>=0.24->segmentation-models-pytorch) (2025.3.2)\n",
            "Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub>=0.24->segmentation-models-pytorch) (24.2)\n",
            "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub>=0.24->segmentation-models-pytorch) (6.0.2)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from huggingface-hub>=0.24->segmentation-models-pytorch) (2.32.3)\n",
            "Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub>=0.24->segmentation-models-pytorch) (4.13.2)\n",
            "Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from torch>=1.8->segmentation-models-pytorch) (3.4.2)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8->segmentation-models-pytorch) (3.1.6)\n",
            "Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8->segmentation-models-pytorch) (12.4.127)\n",
            "Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8->segmentation-models-pytorch) (12.4.127)\n",
            "Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8->segmentation-models-pytorch) (12.4.127)\n",
            "Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8->segmentation-models-pytorch) (9.1.0.70)\n",
            "Requirement already satisfied: nvidia-cublas-cu12==12.4.5.8 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8->segmentation-models-pytorch) (12.4.5.8)\n",
            "Requirement already satisfied: nvidia-cufft-cu12==11.2.1.3 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8->segmentation-models-pytorch) (11.2.1.3)\n",
            "Requirement already satisfied: nvidia-curand-cu12==10.3.5.147 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8->segmentation-models-pytorch) (10.3.5.147)\n",
            "Requirement already satisfied: nvidia-cusolver-cu12==11.6.1.9 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8->segmentation-models-pytorch) (11.6.1.9)\n",
            "Requirement already satisfied: nvidia-cusparse-cu12==12.3.1.170 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8->segmentation-models-pytorch) (12.3.1.170)\n",
            "Requirement already satisfied: nvidia-cusparselt-cu12==0.6.2 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8->segmentation-models-pytorch) (0.6.2)\n",
            "Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8->segmentation-models-pytorch) (2.21.5)\n",
            "Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8->segmentation-models-pytorch) (12.4.127)\n",
            "Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8->segmentation-models-pytorch) (12.4.127)\n",
            "Requirement already satisfied: triton==3.2.0 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8->segmentation-models-pytorch) (3.2.0)\n",
            "Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8->segmentation-models-pytorch) (1.13.1)\n",
            "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from sympy==1.13.1->torch>=1.8->segmentation-models-pytorch) (1.3.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch>=1.8->segmentation-models-pytorch) (3.0.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface-hub>=0.24->segmentation-models-pytorch) (3.4.1)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface-hub>=0.24->segmentation-models-pytorch) (3.7)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface-hub>=0.24->segmentation-models-pytorch) (2.3.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests->huggingface-hub>=0.24->segmentation-models-pytorch) (2025.1.31)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "---\n",
        "\n",
        "#**Custom Dataset Class for Crack Segmentation**\n",
        "\n",
        "---\n",
        "\n"
      ],
      "metadata": {
        "id": "NUbmR6_iMbLf"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "class CrackDataset(Dataset):\n",
        "    def __init__(self, image_dir, mask_dir, transform=None):\n",
        "        self.image_dir = image_dir\n",
        "        self.mask_dir = mask_dir\n",
        "        self.transform = transform\n",
        "\n",
        "        # Create lists of image and mask filenames\n",
        "        self.image_list = sorted(os.listdir(image_dir))\n",
        "        self.mask_list = sorted(os.listdir(mask_dir))\n",
        "\n",
        "        # Debugging: Print the first few filenames\n",
        "        print(\"First 5 images:\", self.image_list[:5])\n",
        "        print(\"First 5 masks:\", self.mask_list[:5])\n",
        "\n",
        "        # Ensure that the lists are not empty\n",
        "        if not self.image_list:\n",
        "            raise ValueError(f\"No images found in {image_dir}\")\n",
        "        if not self.mask_list:\n",
        "            raise ValueError(f\"No masks found in {mask_dir}\")\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.image_list)\n",
        "\n",
        "    def __getitem__(self, idx):\n",
        "        image_path = os.path.join(self.image_dir, self.image_list[idx])\n",
        "        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])\n",
        "\n",
        "        # Debugging: Print paths\n",
        "        #print(\"Image path:\", image_path)\n",
        "        #print(\"Mask path:\", mask_path)\n",
        "\n",
        "        # Load image and mask\n",
        "        image = np.array(Image.open(image_path).convert(\"RGB\"))\n",
        "        mask = np.array(Image.open(mask_path))\n",
        "\n",
        "        # Debugging: Print shapes\n",
        "        #print(\"Image shape:\", image.shape)\n",
        "        #print(\"Mask shape:\", mask.shape)\n",
        "\n",
        "        if self.transform:\n",
        "            augmented = self.transform(image=image, mask=mask)\n",
        "            image = augmented[\"image\"]\n",
        "            mask = augmented[\"mask\"]\n",
        "\n",
        "        mask = torch.from_numpy(np.array(mask)).long()  # make sure it's a tensor with class labels\n",
        "        return image, mask"
      ],
      "metadata": {
        "id": "3HgzKaCmADVI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "---\n",
        "\n",
        "#**Apply Transformations and Create DataLoader**\n",
        "\n",
        "---\n",
        "\n"
      ],
      "metadata": {
        "id": "QUd0cDUeNzU5"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from torch.utils.data import DataLoader\n",
        "import albumentations as A\n",
        "from albumentations.pytorch import ToTensorV2\n",
        "\n",
        "transform = A.Compose([\n",
        "    A.Resize(256, 256),\n",
        "    A.Normalize(mean=(0.5,), std=(0.5,)),  # Or (0.485, 0.456, 0.406) if pretrained model used\n",
        "    ToTensorV2()\n",
        "])\n",
        "\n",
        "train_dataset = CrackDataset(\n",
        "    image_dir='/content/drive/My Drive/CrackDetection/train/images',\n",
        "    mask_dir='/content/drive/My Drive/CrackDetection/train/mkimages',\n",
        "    transform=transform  # same transform from earlier\n",
        ")\n",
        "# Test the dataset\n",
        "image, mask = train_dataset[0]\n",
        "#print(\"Image shape:\", image.shape)\n",
        "#print(\"Mask shape:\", mask.shape)\n",
        "train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gxrjs04eAFHK",
        "outputId": "3d2d093a-8c19-4609-f506-72132b9dbbd6"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "First 5 images: ['-0_2X-16-_jpg.rf.1e2c432ea785a0075abc7b8597810d65.jpg', '-0_2X-16-_jpg.rf.735bff904395fc9a0e21bd5762f1c79e.jpg', '-0_2X-16-_jpg.rf.c40f04b77da485a7b443a327e18dead2.jpg', '-0_2X-163-_jpg.rf.2f71a6676455b5f695371cc680da910b.jpg', '-0_2X-163-_jpg.rf.309422b89778d4a391ffa675a7fea589.jpg']\n",
            "First 5 masks: ['-0_2X-16-_jpg.rf.1e2c432ea785a0075abc7b8597810d65.png', '-0_2X-16-_jpg.rf.735bff904395fc9a0e21bd5762f1c79e.png', '-0_2X-16-_jpg.rf.c40f04b77da485a7b443a327e18dead2.png', '-0_2X-163-_jpg.rf.2f71a6676455b5f695371cc680da910b.png', '-0_2X-163-_jpg.rf.309422b89778d4a391ffa675a7fea589.png']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Installation**"
      ],
      "metadata": {
        "id": "fpWS0kYmOJcZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install -q segmentation-models-pytorch\n",
        "!pip install scikit-learn\n",
        "import segmentation_models_pytorch as smp\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "from sklearn.metrics import precision_score\n",
        "import torch.nn.functional as F"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ryzw2xaLAHHq",
        "outputId": "b7d674fa-cc2e-42e5-c873-2a6788dbe1ea"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: scikit-learn in /usr/local/lib/python3.11/dist-packages (1.6.1)\n",
            "Requirement already satisfied: numpy>=1.19.5 in /usr/local/lib/python3.11/dist-packages (from scikit-learn) (2.0.2)\n",
            "Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn) (1.14.1)\n",
            "Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn) (1.4.2)\n",
            "Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn) (3.6.0)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "---\n",
        "\n",
        "#**Define UNet Model, Loss Function, and Optimizer**\n",
        "\n",
        "---\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "TdqpCSHUOgLw"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Define number of classes\n",
        "NUM_CLASSES = 6  # including background (0)\n",
        "\n",
        "# Model\n",
        "model = smp.Unet(\n",
        "    encoder_name=\"resnet34\",        # pre-trained encoder\n",
        "    encoder_weights=\"imagenet\",\n",
        "    in_channels=3,                  # RGB input\n",
        "    classes=NUM_CLASSES,            # crack classes\n",
        ")\n",
        "\n",
        "# Move model to GPU if available\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "model = model.to(device)\n",
        "\n",
        "# Assign low weight to background (class 0), and higher to crack classes (1–5)\n",
        "class_weights = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float).to(device)\n",
        "\n",
        "loss_fn = nn.CrossEntropyLoss(weight=class_weights)\n",
        "\n",
        "# Optimizer\n",
        "optimizer = optim.Adam(model.parameters(), lr=1e-4)\n"
      ],
      "metadata": {
        "id": "zaQQeCDoAKsv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "---\n",
        "\n",
        "#**Model Training Over 350 Epochs with Progress Tracking**\n",
        "\n",
        "---\n",
        "\n"
      ],
      "metadata": {
        "id": "KjMmGO6LO8yY"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from tqdm import tqdm\n",
        "EPOCHS = 350\n",
        "\n",
        "history = {\n",
        "    \"loss\": [],\n",
        "    \"accuracy\": [],\n",
        "    \"precision\": []\n",
        "}\n",
        "\n",
        "for epoch in range(EPOCHS):\n",
        "    model.train()\n",
        "    running_loss = 0.0\n",
        "    total_correct = 0\n",
        "    total_pixels = 0\n",
        "    total_precision = 0.0\n",
        "\n",
        "    print(f\"\\n🔄 Epoch {epoch+1}/{EPOCHS}\")\n",
        "    progress_bar = tqdm(train_loader, desc=f\"Epoch {epoch+1}/{EPOCHS}\", leave=False)\n",
        "\n",
        "    for images, masks in progress_bar:\n",
        "        images = images.to(device)\n",
        "        masks = masks.to(device)\n",
        "\n",
        "        optimizer.zero_grad()\n",
        "        outputs = model(images)\n",
        "        loss = loss_fn(outputs, masks)\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "\n",
        "        running_loss += loss.item()\n",
        "\n",
        "        # Calculate accuracy\n",
        "        preds = torch.argmax(outputs, dim=1)\n",
        "        total_correct += (preds == masks).sum().item()\n",
        "        total_pixels += torch.numel(masks)\n",
        "\n",
        "        # Calculate precision\n",
        "        preds_flat = preds.view(-1).cpu().numpy()\n",
        "        masks_flat = masks.view(-1).cpu().numpy()\n",
        "        precision = precision_score(masks_flat, preds_flat, average='macro', zero_division=0)\n",
        "        total_precision += precision\n",
        "\n",
        "    # Calculate averages\n",
        "    avg_loss = running_loss / len(train_loader)\n",
        "    avg_accuracy = total_correct / total_pixels\n",
        "    avg_precision = total_precision / len(train_loader)\n",
        "    history[\"loss\"].append(avg_loss)\n",
        "    history[\"accuracy\"].append(avg_accuracy)\n",
        "    history[\"precision\"].append(avg_precision)\n",
        "\n",
        "\n",
        "    # Final summary for the epoch\n",
        "    print(f\"✅ Epoch {epoch+1:>2}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {avg_accuracy:.4f} | Precision: {avg_precision:.4f}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_x2YgVj2H7MJ",
        "outputId": "70f1727a-bc55-4d2e-de09-28914dfd0125"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "🔄 Epoch 1/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch  1/350 | Loss: 1.2903 | Accuracy: 0.7539 | Precision: 0.2381\n",
            "\n",
            "🔄 Epoch 2/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch  2/350 | Loss: 0.4844 | Accuracy: 0.9603 | Precision: 0.3827\n",
            "\n",
            "🔄 Epoch 3/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch  3/350 | Loss: 0.2582 | Accuracy: 0.9680 | Precision: 0.4521\n",
            "\n",
            "🔄 Epoch 4/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch  4/350 | Loss: 0.1778 | Accuracy: 0.9717 | Precision: 0.4842\n",
            "\n",
            "🔄 Epoch 5/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch  5/350 | Loss: 0.1415 | Accuracy: 0.9735 | Precision: 0.4974\n",
            "\n",
            "🔄 Epoch 6/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch  6/350 | Loss: 0.1226 | Accuracy: 0.9746 | Precision: 0.5052\n",
            "\n",
            "🔄 Epoch 7/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch  7/350 | Loss: 0.1098 | Accuracy: 0.9759 | Precision: 0.5183\n",
            "\n",
            "🔄 Epoch 8/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch  8/350 | Loss: 0.1012 | Accuracy: 0.9760 | Precision: 0.5230\n",
            "\n",
            "🔄 Epoch 9/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch  9/350 | Loss: 0.0917 | Accuracy: 0.9771 | Precision: 0.5308\n",
            "\n",
            "🔄 Epoch 10/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 10/350 | Loss: 0.0836 | Accuracy: 0.9781 | Precision: 0.5377\n",
            "\n",
            "🔄 Epoch 11/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 11/350 | Loss: 0.0800 | Accuracy: 0.9789 | Precision: 0.5491\n",
            "\n",
            "🔄 Epoch 12/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 12/350 | Loss: 0.0755 | Accuracy: 0.9793 | Precision: 0.5505\n",
            "\n",
            "🔄 Epoch 13/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 13/350 | Loss: 0.0708 | Accuracy: 0.9803 | Precision: 0.5625\n",
            "\n",
            "🔄 Epoch 14/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 14/350 | Loss: 0.0731 | Accuracy: 0.9802 | Precision: 0.5634\n",
            "\n",
            "🔄 Epoch 15/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 15/350 | Loss: 0.0768 | Accuracy: 0.9791 | Precision: 0.5496\n",
            "\n",
            "🔄 Epoch 16/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 16/350 | Loss: 0.0635 | Accuracy: 0.9814 | Precision: 0.5720\n",
            "\n",
            "🔄 Epoch 17/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 17/350 | Loss: 0.0602 | Accuracy: 0.9820 | Precision: 0.5821\n",
            "\n",
            "🔄 Epoch 18/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 18/350 | Loss: 0.0586 | Accuracy: 0.9826 | Precision: 0.5861\n",
            "\n",
            "🔄 Epoch 19/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 19/350 | Loss: 0.0690 | Accuracy: 0.9804 | Precision: 0.5646\n",
            "\n",
            "🔄 Epoch 20/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 20/350 | Loss: 0.0600 | Accuracy: 0.9819 | Precision: 0.5784\n",
            "\n",
            "🔄 Epoch 21/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 21/350 | Loss: 0.0560 | Accuracy: 0.9827 | Precision: 0.5900\n",
            "\n",
            "🔄 Epoch 22/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 22/350 | Loss: 0.0521 | Accuracy: 0.9837 | Precision: 0.6043\n",
            "\n",
            "🔄 Epoch 23/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 23/350 | Loss: 0.0505 | Accuracy: 0.9841 | Precision: 0.6104\n",
            "\n",
            "🔄 Epoch 24/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 24/350 | Loss: 0.0506 | Accuracy: 0.9844 | Precision: 0.6079\n",
            "\n",
            "🔄 Epoch 25/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 25/350 | Loss: 0.0494 | Accuracy: 0.9844 | Precision: 0.6123\n",
            "\n",
            "🔄 Epoch 26/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 26/350 | Loss: 0.0486 | Accuracy: 0.9847 | Precision: 0.6117\n",
            "\n",
            "🔄 Epoch 27/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 27/350 | Loss: 0.0462 | Accuracy: 0.9853 | Precision: 0.6259\n",
            "\n",
            "🔄 Epoch 28/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 28/350 | Loss: 0.0498 | Accuracy: 0.9846 | Precision: 0.6144\n",
            "\n",
            "🔄 Epoch 29/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 29/350 | Loss: 0.0490 | Accuracy: 0.9848 | Precision: 0.6154\n",
            "\n",
            "🔄 Epoch 30/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 30/350 | Loss: 0.0434 | Accuracy: 0.9860 | Precision: 0.6351\n",
            "\n",
            "🔄 Epoch 31/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 31/350 | Loss: 0.0410 | Accuracy: 0.9866 | Precision: 0.6440\n",
            "\n",
            "🔄 Epoch 32/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 32/350 | Loss: 0.0401 | Accuracy: 0.9869 | Precision: 0.6443\n",
            "\n",
            "🔄 Epoch 33/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 33/350 | Loss: 0.0399 | Accuracy: 0.9870 | Precision: 0.6496\n",
            "\n",
            "🔄 Epoch 34/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 34/350 | Loss: 0.0389 | Accuracy: 0.9872 | Precision: 0.6567\n",
            "\n",
            "🔄 Epoch 35/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 35/350 | Loss: 0.0396 | Accuracy: 0.9872 | Precision: 0.6553\n",
            "\n",
            "🔄 Epoch 36/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 36/350 | Loss: 0.0410 | Accuracy: 0.9869 | Precision: 0.6503\n",
            "\n",
            "🔄 Epoch 37/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 37/350 | Loss: 0.0384 | Accuracy: 0.9875 | Precision: 0.6558\n",
            "\n",
            "🔄 Epoch 38/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 38/350 | Loss: 0.0552 | Accuracy: 0.9852 | Precision: 0.6241\n",
            "\n",
            "🔄 Epoch 39/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 39/350 | Loss: 0.0821 | Accuracy: 0.9784 | Precision: 0.5435\n",
            "\n",
            "🔄 Epoch 40/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 40/350 | Loss: 0.0478 | Accuracy: 0.9847 | Precision: 0.6143\n",
            "\n",
            "🔄 Epoch 41/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 41/350 | Loss: 0.0429 | Accuracy: 0.9861 | Precision: 0.6342\n",
            "\n",
            "🔄 Epoch 42/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 42/350 | Loss: 0.0404 | Accuracy: 0.9868 | Precision: 0.6407\n",
            "\n",
            "🔄 Epoch 43/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 43/350 | Loss: 0.0371 | Accuracy: 0.9877 | Precision: 0.6555\n",
            "\n",
            "🔄 Epoch 44/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 44/350 | Loss: 0.0355 | Accuracy: 0.9882 | Precision: 0.6672\n",
            "\n",
            "🔄 Epoch 45/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 45/350 | Loss: 0.0347 | Accuracy: 0.9884 | Precision: 0.6737\n",
            "\n",
            "🔄 Epoch 46/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 46/350 | Loss: 0.0334 | Accuracy: 0.9888 | Precision: 0.6755\n",
            "\n",
            "🔄 Epoch 47/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 47/350 | Loss: 0.0329 | Accuracy: 0.9890 | Precision: 0.6846\n",
            "\n",
            "🔄 Epoch 48/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 48/350 | Loss: 0.0324 | Accuracy: 0.9892 | Precision: 0.6840\n",
            "\n",
            "🔄 Epoch 49/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 49/350 | Loss: 0.0325 | Accuracy: 0.9891 | Precision: 0.6847\n",
            "\n",
            "🔄 Epoch 50/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 50/350 | Loss: 0.0334 | Accuracy: 0.9890 | Precision: 0.6784\n",
            "\n",
            "🔄 Epoch 51/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 51/350 | Loss: 0.0315 | Accuracy: 0.9895 | Precision: 0.6903\n",
            "\n",
            "🔄 Epoch 52/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 52/350 | Loss: 0.0321 | Accuracy: 0.9893 | Precision: 0.6878\n",
            "\n",
            "🔄 Epoch 53/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 53/350 | Loss: 0.0305 | Accuracy: 0.9897 | Precision: 0.6976\n",
            "\n",
            "🔄 Epoch 54/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 54/350 | Loss: 0.0306 | Accuracy: 0.9898 | Precision: 0.6932\n",
            "\n",
            "🔄 Epoch 55/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 55/350 | Loss: 0.0304 | Accuracy: 0.9898 | Precision: 0.6985\n",
            "\n",
            "🔄 Epoch 56/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 56/350 | Loss: 0.0306 | Accuracy: 0.9898 | Precision: 0.6968\n",
            "\n",
            "🔄 Epoch 57/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 57/350 | Loss: 0.0297 | Accuracy: 0.9901 | Precision: 0.7003\n",
            "\n",
            "🔄 Epoch 58/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 58/350 | Loss: 0.0296 | Accuracy: 0.9902 | Precision: 0.7012\n",
            "\n",
            "🔄 Epoch 59/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 59/350 | Loss: 0.0293 | Accuracy: 0.9903 | Precision: 0.7044\n",
            "\n",
            "🔄 Epoch 60/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 60/350 | Loss: 0.0291 | Accuracy: 0.9903 | Precision: 0.7064\n",
            "\n",
            "🔄 Epoch 61/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 61/350 | Loss: 0.0286 | Accuracy: 0.9905 | Precision: 0.7135\n",
            "\n",
            "🔄 Epoch 62/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 62/350 | Loss: 0.0288 | Accuracy: 0.9904 | Precision: 0.7086\n",
            "\n",
            "🔄 Epoch 63/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 63/350 | Loss: 0.0278 | Accuracy: 0.9907 | Precision: 0.7116\n",
            "\n",
            "🔄 Epoch 64/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 64/350 | Loss: 0.0277 | Accuracy: 0.9907 | Precision: 0.7171\n",
            "\n",
            "🔄 Epoch 65/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 65/350 | Loss: 0.0278 | Accuracy: 0.9907 | Precision: 0.7145\n",
            "\n",
            "🔄 Epoch 66/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 66/350 | Loss: 0.0271 | Accuracy: 0.9909 | Precision: 0.7210\n",
            "\n",
            "🔄 Epoch 67/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 67/350 | Loss: 0.0265 | Accuracy: 0.9911 | Precision: 0.7253\n",
            "\n",
            "🔄 Epoch 68/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 68/350 | Loss: 0.0264 | Accuracy: 0.9912 | Precision: 0.7243\n",
            "\n",
            "🔄 Epoch 69/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 69/350 | Loss: 0.0368 | Accuracy: 0.9895 | Precision: 0.6880\n",
            "\n",
            "🔄 Epoch 70/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 70/350 | Loss: 0.0492 | Accuracy: 0.9854 | Precision: 0.6196\n",
            "\n",
            "🔄 Epoch 71/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 71/350 | Loss: 0.0319 | Accuracy: 0.9895 | Precision: 0.6909\n",
            "\n",
            "🔄 Epoch 72/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 72/350 | Loss: 0.0273 | Accuracy: 0.9908 | Precision: 0.7144\n",
            "\n",
            "🔄 Epoch 73/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 73/350 | Loss: 0.0260 | Accuracy: 0.9913 | Precision: 0.7250\n",
            "\n",
            "🔄 Epoch 74/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 74/350 | Loss: 0.0249 | Accuracy: 0.9917 | Precision: 0.7363\n",
            "\n",
            "🔄 Epoch 75/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 75/350 | Loss: 0.0255 | Accuracy: 0.9916 | Precision: 0.7326\n",
            "\n",
            "🔄 Epoch 76/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 76/350 | Loss: 0.0319 | Accuracy: 0.9900 | Precision: 0.7020\n",
            "\n",
            "🔄 Epoch 77/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 77/350 | Loss: 0.0253 | Accuracy: 0.9915 | Precision: 0.7319\n",
            "\n",
            "🔄 Epoch 78/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 78/350 | Loss: 0.0243 | Accuracy: 0.9919 | Precision: 0.7403\n",
            "\n",
            "🔄 Epoch 79/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 79/350 | Loss: 0.0245 | Accuracy: 0.9918 | Precision: 0.7359\n",
            "\n",
            "🔄 Epoch 80/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 80/350 | Loss: 0.0232 | Accuracy: 0.9922 | Precision: 0.7472\n",
            "\n",
            "🔄 Epoch 81/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 81/350 | Loss: 0.0228 | Accuracy: 0.9924 | Precision: 0.7518\n",
            "\n",
            "🔄 Epoch 82/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 82/350 | Loss: 0.0227 | Accuracy: 0.9924 | Precision: 0.7506\n",
            "\n",
            "🔄 Epoch 83/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 83/350 | Loss: 0.0232 | Accuracy: 0.9923 | Precision: 0.7505\n",
            "\n",
            "🔄 Epoch 84/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 84/350 | Loss: 0.0253 | Accuracy: 0.9916 | Precision: 0.7308\n",
            "\n",
            "🔄 Epoch 85/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 85/350 | Loss: 0.0250 | Accuracy: 0.9918 | Precision: 0.7366\n",
            "\n",
            "🔄 Epoch 86/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 86/350 | Loss: 0.0274 | Accuracy: 0.9912 | Precision: 0.7222\n",
            "\n",
            "🔄 Epoch 87/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 87/350 | Loss: 0.0242 | Accuracy: 0.9919 | Precision: 0.7381\n",
            "\n",
            "🔄 Epoch 88/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 88/350 | Loss: 0.0230 | Accuracy: 0.9925 | Precision: 0.7507\n",
            "\n",
            "🔄 Epoch 89/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 89/350 | Loss: 0.0339 | Accuracy: 0.9898 | Precision: 0.6853\n",
            "\n",
            "🔄 Epoch 90/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 90/350 | Loss: 0.0292 | Accuracy: 0.9904 | Precision: 0.7014\n",
            "\n",
            "🔄 Epoch 91/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 91/350 | Loss: 0.0247 | Accuracy: 0.9919 | Precision: 0.7363\n",
            "\n",
            "🔄 Epoch 92/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 92/350 | Loss: 0.0218 | Accuracy: 0.9927 | Precision: 0.7518\n",
            "\n",
            "🔄 Epoch 93/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 93/350 | Loss: 0.0209 | Accuracy: 0.9930 | Precision: 0.7655\n",
            "\n",
            "🔄 Epoch 94/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 94/350 | Loss: 0.0209 | Accuracy: 0.9931 | Precision: 0.7654\n",
            "\n",
            "🔄 Epoch 95/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 95/350 | Loss: 0.0228 | Accuracy: 0.9931 | Precision: 0.7609\n",
            "\n",
            "🔄 Epoch 96/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 96/350 | Loss: 0.0290 | Accuracy: 0.9907 | Precision: 0.7099\n",
            "\n",
            "🔄 Epoch 97/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 97/350 | Loss: 0.0223 | Accuracy: 0.9926 | Precision: 0.7490\n",
            "\n",
            "🔄 Epoch 98/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 98/350 | Loss: 0.0217 | Accuracy: 0.9929 | Precision: 0.7601\n",
            "\n",
            "🔄 Epoch 99/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 99/350 | Loss: 0.0201 | Accuracy: 0.9933 | Precision: 0.7725\n",
            "\n",
            "🔄 Epoch 100/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 100/350 | Loss: 0.0198 | Accuracy: 0.9934 | Precision: 0.7735\n",
            "\n",
            "🔄 Epoch 101/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 101/350 | Loss: 0.0197 | Accuracy: 0.9935 | Precision: 0.7811\n",
            "\n",
            "🔄 Epoch 102/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 102/350 | Loss: 0.0196 | Accuracy: 0.9935 | Precision: 0.7793\n",
            "\n",
            "🔄 Epoch 103/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 103/350 | Loss: 0.0191 | Accuracy: 0.9937 | Precision: 0.7807\n",
            "\n",
            "🔄 Epoch 104/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 104/350 | Loss: 0.0192 | Accuracy: 0.9937 | Precision: 0.7815\n",
            "\n",
            "🔄 Epoch 105/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 105/350 | Loss: 0.0277 | Accuracy: 0.9915 | Precision: 0.7305\n",
            "\n",
            "🔄 Epoch 106/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 106/350 | Loss: 0.0216 | Accuracy: 0.9929 | Precision: 0.7554\n",
            "\n",
            "🔄 Epoch 107/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 107/350 | Loss: 0.0201 | Accuracy: 0.9933 | Precision: 0.7721\n",
            "\n",
            "🔄 Epoch 108/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 108/350 | Loss: 0.0190 | Accuracy: 0.9937 | Precision: 0.7824\n",
            "\n",
            "🔄 Epoch 109/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 109/350 | Loss: 0.0187 | Accuracy: 0.9938 | Precision: 0.7859\n",
            "\n",
            "🔄 Epoch 110/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 110/350 | Loss: 0.0190 | Accuracy: 0.9938 | Precision: 0.7821\n",
            "\n",
            "🔄 Epoch 111/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 111/350 | Loss: 0.0192 | Accuracy: 0.9937 | Precision: 0.7779\n",
            "\n",
            "🔄 Epoch 112/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 112/350 | Loss: 0.0194 | Accuracy: 0.9936 | Precision: 0.7795\n",
            "\n",
            "🔄 Epoch 113/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 113/350 | Loss: 0.0194 | Accuracy: 0.9937 | Precision: 0.7837\n",
            "\n",
            "🔄 Epoch 114/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 114/350 | Loss: 0.0214 | Accuracy: 0.9930 | Precision: 0.7651\n",
            "\n",
            "🔄 Epoch 115/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 115/350 | Loss: 0.0204 | Accuracy: 0.9933 | Precision: 0.7724\n",
            "\n",
            "🔄 Epoch 116/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 116/350 | Loss: 0.0184 | Accuracy: 0.9939 | Precision: 0.7866\n",
            "\n",
            "🔄 Epoch 117/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 117/350 | Loss: 0.0181 | Accuracy: 0.9940 | Precision: 0.7901\n",
            "\n",
            "🔄 Epoch 118/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 118/350 | Loss: 0.0180 | Accuracy: 0.9941 | Precision: 0.7923\n",
            "\n",
            "🔄 Epoch 119/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 119/350 | Loss: 0.0186 | Accuracy: 0.9940 | Precision: 0.7876\n",
            "\n",
            "🔄 Epoch 120/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 120/350 | Loss: 0.0190 | Accuracy: 0.9938 | Precision: 0.7818\n",
            "\n",
            "🔄 Epoch 121/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 121/350 | Loss: 0.0291 | Accuracy: 0.9915 | Precision: 0.7303\n",
            "\n",
            "🔄 Epoch 122/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 122/350 | Loss: 0.0363 | Accuracy: 0.9887 | Precision: 0.6736\n",
            "\n",
            "🔄 Epoch 123/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 123/350 | Loss: 0.0229 | Accuracy: 0.9922 | Precision: 0.7439\n",
            "\n",
            "🔄 Epoch 124/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 124/350 | Loss: 0.0197 | Accuracy: 0.9934 | Precision: 0.7712\n",
            "\n",
            "🔄 Epoch 125/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 125/350 | Loss: 0.0185 | Accuracy: 0.9938 | Precision: 0.7832\n",
            "\n",
            "🔄 Epoch 126/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 126/350 | Loss: 0.0207 | Accuracy: 0.9934 | Precision: 0.7695\n",
            "\n",
            "🔄 Epoch 127/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 127/350 | Loss: 0.0193 | Accuracy: 0.9936 | Precision: 0.7750\n",
            "\n",
            "🔄 Epoch 128/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 128/350 | Loss: 0.0173 | Accuracy: 0.9942 | Precision: 0.7909\n",
            "\n",
            "🔄 Epoch 129/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 129/350 | Loss: 0.0165 | Accuracy: 0.9946 | Precision: 0.8032\n",
            "\n",
            "🔄 Epoch 130/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 130/350 | Loss: 0.0165 | Accuracy: 0.9946 | Precision: 0.8000\n",
            "\n",
            "🔄 Epoch 131/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 131/350 | Loss: 0.0161 | Accuracy: 0.9947 | Precision: 0.8077\n",
            "\n",
            "🔄 Epoch 132/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 132/350 | Loss: 0.0161 | Accuracy: 0.9947 | Precision: 0.8069\n",
            "\n",
            "🔄 Epoch 133/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 133/350 | Loss: 0.0159 | Accuracy: 0.9948 | Precision: 0.8101\n",
            "\n",
            "🔄 Epoch 134/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 134/350 | Loss: 0.0157 | Accuracy: 0.9949 | Precision: 0.8114\n",
            "\n",
            "🔄 Epoch 135/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 135/350 | Loss: 0.0162 | Accuracy: 0.9947 | Precision: 0.8101\n",
            "\n",
            "🔄 Epoch 136/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 136/350 | Loss: 0.0164 | Accuracy: 0.9947 | Precision: 0.8062\n",
            "\n",
            "🔄 Epoch 137/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 137/350 | Loss: 0.0161 | Accuracy: 0.9948 | Precision: 0.8073\n",
            "\n",
            "🔄 Epoch 138/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 138/350 | Loss: 0.0158 | Accuracy: 0.9948 | Precision: 0.8099\n",
            "\n",
            "🔄 Epoch 139/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 139/350 | Loss: 0.0183 | Accuracy: 0.9943 | Precision: 0.8024\n",
            "\n",
            "🔄 Epoch 140/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 140/350 | Loss: 0.0198 | Accuracy: 0.9936 | Precision: 0.7845\n",
            "\n",
            "🔄 Epoch 141/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 141/350 | Loss: 0.0179 | Accuracy: 0.9942 | Precision: 0.7963\n",
            "\n",
            "🔄 Epoch 142/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 142/350 | Loss: 0.0164 | Accuracy: 0.9947 | Precision: 0.8041\n",
            "\n",
            "🔄 Epoch 143/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 143/350 | Loss: 0.0156 | Accuracy: 0.9949 | Precision: 0.8085\n",
            "\n",
            "🔄 Epoch 144/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 144/350 | Loss: 0.0157 | Accuracy: 0.9949 | Precision: 0.8124\n",
            "\n",
            "🔄 Epoch 145/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 145/350 | Loss: 0.0184 | Accuracy: 0.9943 | Precision: 0.7980\n",
            "\n",
            "🔄 Epoch 146/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 146/350 | Loss: 0.0203 | Accuracy: 0.9935 | Precision: 0.7769\n",
            "\n",
            "🔄 Epoch 147/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 147/350 | Loss: 0.0169 | Accuracy: 0.9945 | Precision: 0.7999\n",
            "\n",
            "🔄 Epoch 148/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 148/350 | Loss: 0.0171 | Accuracy: 0.9945 | Precision: 0.7991\n",
            "\n",
            "🔄 Epoch 149/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 149/350 | Loss: 0.0152 | Accuracy: 0.9950 | Precision: 0.8121\n",
            "\n",
            "🔄 Epoch 150/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 150/350 | Loss: 0.0151 | Accuracy: 0.9951 | Precision: 0.8183\n",
            "\n",
            "🔄 Epoch 151/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 151/350 | Loss: 0.0153 | Accuracy: 0.9951 | Precision: 0.8198\n",
            "\n",
            "🔄 Epoch 152/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 152/350 | Loss: 0.0151 | Accuracy: 0.9951 | Precision: 0.8183\n",
            "\n",
            "🔄 Epoch 153/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 153/350 | Loss: 0.0156 | Accuracy: 0.9950 | Precision: 0.8180\n",
            "\n",
            "🔄 Epoch 154/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 154/350 | Loss: 0.0160 | Accuracy: 0.9948 | Precision: 0.8097\n",
            "\n",
            "🔄 Epoch 155/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 155/350 | Loss: 0.0150 | Accuracy: 0.9952 | Precision: 0.8164\n",
            "\n",
            "🔄 Epoch 156/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 156/350 | Loss: 0.0158 | Accuracy: 0.9949 | Precision: 0.8088\n",
            "\n",
            "🔄 Epoch 157/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 157/350 | Loss: 0.0154 | Accuracy: 0.9950 | Precision: 0.8135\n",
            "\n",
            "🔄 Epoch 158/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 158/350 | Loss: 0.0149 | Accuracy: 0.9952 | Precision: 0.8194\n",
            "\n",
            "🔄 Epoch 159/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 159/350 | Loss: 0.0145 | Accuracy: 0.9953 | Precision: 0.8209\n",
            "\n",
            "🔄 Epoch 160/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 160/350 | Loss: 0.0144 | Accuracy: 0.9953 | Precision: 0.8228\n",
            "\n",
            "🔄 Epoch 161/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 161/350 | Loss: 0.0143 | Accuracy: 0.9954 | Precision: 0.8217\n",
            "\n",
            "🔄 Epoch 162/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 162/350 | Loss: 0.0144 | Accuracy: 0.9954 | Precision: 0.8272\n",
            "\n",
            "🔄 Epoch 163/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 163/350 | Loss: 0.0156 | Accuracy: 0.9951 | Precision: 0.8184\n",
            "\n",
            "🔄 Epoch 164/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 164/350 | Loss: 0.0147 | Accuracy: 0.9953 | Precision: 0.8224\n",
            "\n",
            "🔄 Epoch 165/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 165/350 | Loss: 0.0150 | Accuracy: 0.9952 | Precision: 0.8215\n",
            "\n",
            "🔄 Epoch 166/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 166/350 | Loss: 0.0146 | Accuracy: 0.9953 | Precision: 0.8246\n",
            "\n",
            "🔄 Epoch 167/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 167/350 | Loss: 0.0145 | Accuracy: 0.9954 | Precision: 0.8248\n",
            "\n",
            "🔄 Epoch 168/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 168/350 | Loss: 0.0141 | Accuracy: 0.9955 | Precision: 0.8274\n",
            "\n",
            "🔄 Epoch 169/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 169/350 | Loss: 0.0145 | Accuracy: 0.9954 | Precision: 0.8259\n",
            "\n",
            "🔄 Epoch 170/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 170/350 | Loss: 0.0139 | Accuracy: 0.9956 | Precision: 0.8299\n",
            "\n",
            "🔄 Epoch 171/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 171/350 | Loss: 0.0135 | Accuracy: 0.9956 | Precision: 0.8308\n",
            "\n",
            "🔄 Epoch 172/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 172/350 | Loss: 0.0135 | Accuracy: 0.9957 | Precision: 0.8356\n",
            "\n",
            "🔄 Epoch 173/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 173/350 | Loss: 0.0143 | Accuracy: 0.9954 | Precision: 0.8247\n",
            "\n",
            "🔄 Epoch 174/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 174/350 | Loss: 0.0134 | Accuracy: 0.9957 | Precision: 0.8374\n",
            "\n",
            "🔄 Epoch 175/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 175/350 | Loss: 0.0138 | Accuracy: 0.9956 | Precision: 0.8308\n",
            "\n",
            "🔄 Epoch 176/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 176/350 | Loss: 0.0141 | Accuracy: 0.9956 | Precision: 0.8276\n",
            "\n",
            "🔄 Epoch 177/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 177/350 | Loss: 0.0136 | Accuracy: 0.9957 | Precision: 0.8323\n",
            "\n",
            "🔄 Epoch 178/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 178/350 | Loss: 0.0141 | Accuracy: 0.9956 | Precision: 0.8299\n",
            "\n",
            "🔄 Epoch 179/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 179/350 | Loss: 0.0172 | Accuracy: 0.9946 | Precision: 0.8022\n",
            "\n",
            "🔄 Epoch 180/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 180/350 | Loss: 0.0149 | Accuracy: 0.9952 | Precision: 0.8161\n",
            "\n",
            "🔄 Epoch 181/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 181/350 | Loss: 0.0128 | Accuracy: 0.9958 | Precision: 0.8430\n",
            "\n",
            "🔄 Epoch 182/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 182/350 | Loss: 0.0120 | Accuracy: 0.9961 | Precision: 0.8502\n",
            "\n",
            "🔄 Epoch 183/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 183/350 | Loss: 0.0116 | Accuracy: 0.9963 | Precision: 0.8489\n",
            "\n",
            "🔄 Epoch 184/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 184/350 | Loss: 0.0122 | Accuracy: 0.9961 | Precision: 0.8517\n",
            "\n",
            "🔄 Epoch 185/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 185/350 | Loss: 0.0130 | Accuracy: 0.9958 | Precision: 0.8412\n",
            "\n",
            "🔄 Epoch 186/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 186/350 | Loss: 0.0123 | Accuracy: 0.9961 | Precision: 0.8461\n",
            "\n",
            "🔄 Epoch 187/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 187/350 | Loss: 0.0125 | Accuracy: 0.9960 | Precision: 0.8413\n",
            "\n",
            "🔄 Epoch 188/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 188/350 | Loss: 0.0120 | Accuracy: 0.9962 | Precision: 0.8500\n",
            "\n",
            "🔄 Epoch 189/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 189/350 | Loss: 0.0142 | Accuracy: 0.9957 | Precision: 0.8409\n",
            "\n",
            "🔄 Epoch 190/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 190/350 | Loss: 0.0141 | Accuracy: 0.9955 | Precision: 0.8358\n",
            "\n",
            "🔄 Epoch 191/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 191/350 | Loss: 0.0123 | Accuracy: 0.9961 | Precision: 0.8491\n",
            "\n",
            "🔄 Epoch 192/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 192/350 | Loss: 0.0121 | Accuracy: 0.9962 | Precision: 0.8504\n",
            "\n",
            "🔄 Epoch 193/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 193/350 | Loss: 0.0125 | Accuracy: 0.9960 | Precision: 0.8479\n",
            "\n",
            "🔄 Epoch 194/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 194/350 | Loss: 0.0119 | Accuracy: 0.9962 | Precision: 0.8532\n",
            "\n",
            "🔄 Epoch 195/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 195/350 | Loss: 0.0116 | Accuracy: 0.9963 | Precision: 0.8513\n",
            "\n",
            "🔄 Epoch 196/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 196/350 | Loss: 0.0115 | Accuracy: 0.9964 | Precision: 0.8536\n",
            "\n",
            "🔄 Epoch 197/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 197/350 | Loss: 0.0126 | Accuracy: 0.9960 | Precision: 0.8495\n",
            "\n",
            "🔄 Epoch 198/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 198/350 | Loss: 0.0120 | Accuracy: 0.9962 | Precision: 0.8520\n",
            "\n",
            "🔄 Epoch 199/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 199/350 | Loss: 0.0113 | Accuracy: 0.9964 | Precision: 0.8554\n",
            "\n",
            "🔄 Epoch 200/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 200/350 | Loss: 0.0112 | Accuracy: 0.9964 | Precision: 0.8565\n",
            "\n",
            "🔄 Epoch 201/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 201/350 | Loss: 0.0110 | Accuracy: 0.9965 | Precision: 0.8588\n",
            "\n",
            "🔄 Epoch 202/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 202/350 | Loss: 0.0122 | Accuracy: 0.9962 | Precision: 0.8517\n",
            "\n",
            "🔄 Epoch 203/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 203/350 | Loss: 0.0131 | Accuracy: 0.9960 | Precision: 0.8387\n",
            "\n",
            "🔄 Epoch 204/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 204/350 | Loss: 0.0115 | Accuracy: 0.9964 | Precision: 0.8540\n",
            "\n",
            "🔄 Epoch 205/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 205/350 | Loss: 0.0110 | Accuracy: 0.9965 | Precision: 0.8558\n",
            "\n",
            "🔄 Epoch 206/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 206/350 | Loss: 0.0146 | Accuracy: 0.9957 | Precision: 0.8332\n",
            "\n",
            "🔄 Epoch 207/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 207/350 | Loss: 0.0134 | Accuracy: 0.9959 | Precision: 0.8390\n",
            "\n",
            "🔄 Epoch 208/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 208/350 | Loss: 0.0132 | Accuracy: 0.9959 | Precision: 0.8352\n",
            "\n",
            "🔄 Epoch 209/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 209/350 | Loss: 0.0110 | Accuracy: 0.9965 | Precision: 0.8566\n",
            "\n",
            "🔄 Epoch 210/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 210/350 | Loss: 0.0123 | Accuracy: 0.9962 | Precision: 0.8482\n",
            "\n",
            "🔄 Epoch 211/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 211/350 | Loss: 0.0132 | Accuracy: 0.9958 | Precision: 0.8420\n",
            "\n",
            "🔄 Epoch 212/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 212/350 | Loss: 0.0108 | Accuracy: 0.9965 | Precision: 0.8590\n",
            "\n",
            "🔄 Epoch 213/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 213/350 | Loss: 0.0099 | Accuracy: 0.9968 | Precision: 0.8686\n",
            "\n",
            "🔄 Epoch 214/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 214/350 | Loss: 0.0097 | Accuracy: 0.9969 | Precision: 0.8705\n",
            "\n",
            "🔄 Epoch 215/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 215/350 | Loss: 0.0098 | Accuracy: 0.9969 | Precision: 0.8714\n",
            "\n",
            "🔄 Epoch 216/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 216/350 | Loss: 0.0114 | Accuracy: 0.9965 | Precision: 0.8624\n",
            "\n",
            "🔄 Epoch 217/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 217/350 | Loss: 0.0103 | Accuracy: 0.9968 | Precision: 0.8669\n",
            "\n",
            "🔄 Epoch 218/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 218/350 | Loss: 0.0102 | Accuracy: 0.9968 | Precision: 0.8677\n",
            "\n",
            "🔄 Epoch 219/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 219/350 | Loss: 0.0102 | Accuracy: 0.9967 | Precision: 0.8677\n",
            "\n",
            "🔄 Epoch 220/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 220/350 | Loss: 0.0106 | Accuracy: 0.9967 | Precision: 0.8616\n",
            "\n",
            "🔄 Epoch 221/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 221/350 | Loss: 0.0108 | Accuracy: 0.9966 | Precision: 0.8637\n",
            "\n",
            "🔄 Epoch 222/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 222/350 | Loss: 0.0106 | Accuracy: 0.9967 | Precision: 0.8647\n",
            "\n",
            "🔄 Epoch 223/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 223/350 | Loss: 0.0102 | Accuracy: 0.9968 | Precision: 0.8662\n",
            "\n",
            "🔄 Epoch 224/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 224/350 | Loss: 0.0104 | Accuracy: 0.9968 | Precision: 0.8691\n",
            "\n",
            "🔄 Epoch 225/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 225/350 | Loss: 0.0101 | Accuracy: 0.9968 | Precision: 0.8659\n",
            "\n",
            "🔄 Epoch 226/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 226/350 | Loss: 0.0099 | Accuracy: 0.9969 | Precision: 0.8647\n",
            "\n",
            "🔄 Epoch 227/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 227/350 | Loss: 0.0100 | Accuracy: 0.9969 | Precision: 0.8680\n",
            "\n",
            "🔄 Epoch 228/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 228/350 | Loss: 0.0099 | Accuracy: 0.9969 | Precision: 0.8712\n",
            "\n",
            "🔄 Epoch 229/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 229/350 | Loss: 0.0103 | Accuracy: 0.9968 | Precision: 0.8715\n",
            "\n",
            "🔄 Epoch 230/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 230/350 | Loss: 0.0100 | Accuracy: 0.9969 | Precision: 0.8734\n",
            "\n",
            "🔄 Epoch 231/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 231/350 | Loss: 0.0103 | Accuracy: 0.9968 | Precision: 0.8690\n",
            "\n",
            "🔄 Epoch 232/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 232/350 | Loss: 0.0100 | Accuracy: 0.9969 | Precision: 0.8704\n",
            "\n",
            "🔄 Epoch 233/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 233/350 | Loss: 0.0097 | Accuracy: 0.9970 | Precision: 0.8736\n",
            "\n",
            "🔄 Epoch 234/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 234/350 | Loss: 0.0092 | Accuracy: 0.9971 | Precision: 0.8788\n",
            "\n",
            "🔄 Epoch 235/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 235/350 | Loss: 0.0101 | Accuracy: 0.9969 | Precision: 0.8703\n",
            "\n",
            "🔄 Epoch 236/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 236/350 | Loss: 0.0098 | Accuracy: 0.9969 | Precision: 0.8745\n",
            "\n",
            "🔄 Epoch 237/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 237/350 | Loss: 0.0097 | Accuracy: 0.9970 | Precision: 0.8772\n",
            "\n",
            "🔄 Epoch 238/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 238/350 | Loss: 0.0095 | Accuracy: 0.9970 | Precision: 0.8771\n",
            "\n",
            "🔄 Epoch 239/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 239/350 | Loss: 0.0091 | Accuracy: 0.9971 | Precision: 0.8788\n",
            "\n",
            "🔄 Epoch 240/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 240/350 | Loss: 0.0093 | Accuracy: 0.9971 | Precision: 0.8765\n",
            "\n",
            "🔄 Epoch 241/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 241/350 | Loss: 0.0093 | Accuracy: 0.9971 | Precision: 0.8802\n",
            "\n",
            "🔄 Epoch 242/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 242/350 | Loss: 0.0095 | Accuracy: 0.9970 | Precision: 0.8761\n",
            "\n",
            "🔄 Epoch 243/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 243/350 | Loss: 0.0100 | Accuracy: 0.9969 | Precision: 0.8726\n",
            "\n",
            "🔄 Epoch 244/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 244/350 | Loss: 0.0111 | Accuracy: 0.9966 | Precision: 0.8612\n",
            "\n",
            "🔄 Epoch 245/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 245/350 | Loss: 0.0106 | Accuracy: 0.9967 | Precision: 0.8675\n",
            "\n",
            "🔄 Epoch 246/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 246/350 | Loss: 0.0091 | Accuracy: 0.9972 | Precision: 0.8798\n",
            "\n",
            "🔄 Epoch 247/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 247/350 | Loss: 0.0087 | Accuracy: 0.9973 | Precision: 0.8826\n",
            "\n",
            "🔄 Epoch 248/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 248/350 | Loss: 0.0082 | Accuracy: 0.9974 | Precision: 0.8911\n",
            "\n",
            "🔄 Epoch 249/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 249/350 | Loss: 0.0081 | Accuracy: 0.9975 | Precision: 0.8911\n",
            "\n",
            "🔄 Epoch 250/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 250/350 | Loss: 0.0083 | Accuracy: 0.9974 | Precision: 0.8877\n",
            "\n",
            "🔄 Epoch 251/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 251/350 | Loss: 0.0088 | Accuracy: 0.9973 | Precision: 0.8867\n",
            "\n",
            "🔄 Epoch 252/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 252/350 | Loss: 0.0092 | Accuracy: 0.9972 | Precision: 0.8829\n",
            "\n",
            "🔄 Epoch 253/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 253/350 | Loss: 0.0100 | Accuracy: 0.9970 | Precision: 0.8735\n",
            "\n",
            "🔄 Epoch 254/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 254/350 | Loss: 0.0105 | Accuracy: 0.9968 | Precision: 0.8658\n",
            "\n",
            "🔄 Epoch 255/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 255/350 | Loss: 0.0100 | Accuracy: 0.9969 | Precision: 0.8704\n",
            "\n",
            "🔄 Epoch 256/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 256/350 | Loss: 0.0089 | Accuracy: 0.9972 | Precision: 0.8837\n",
            "\n",
            "🔄 Epoch 257/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 257/350 | Loss: 0.0081 | Accuracy: 0.9975 | Precision: 0.8913\n",
            "\n",
            "🔄 Epoch 258/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 258/350 | Loss: 0.0081 | Accuracy: 0.9975 | Precision: 0.8927\n",
            "\n",
            "🔄 Epoch 259/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 259/350 | Loss: 0.0101 | Accuracy: 0.9972 | Precision: 0.8841\n",
            "\n",
            "🔄 Epoch 260/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 260/350 | Loss: 0.0149 | Accuracy: 0.9958 | Precision: 0.8313\n",
            "\n",
            "🔄 Epoch 261/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 261/350 | Loss: 0.0102 | Accuracy: 0.9968 | Precision: 0.8659\n",
            "\n",
            "🔄 Epoch 262/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 262/350 | Loss: 0.0095 | Accuracy: 0.9971 | Precision: 0.8768\n",
            "\n",
            "🔄 Epoch 263/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 263/350 | Loss: 0.0090 | Accuracy: 0.9973 | Precision: 0.8856\n",
            "\n",
            "🔄 Epoch 264/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 264/350 | Loss: 0.0084 | Accuracy: 0.9974 | Precision: 0.8908\n",
            "\n",
            "🔄 Epoch 265/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 265/350 | Loss: 0.0074 | Accuracy: 0.9977 | Precision: 0.9010\n",
            "\n",
            "🔄 Epoch 266/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 266/350 | Loss: 0.0071 | Accuracy: 0.9978 | Precision: 0.9020\n",
            "\n",
            "🔄 Epoch 267/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 267/350 | Loss: 0.0073 | Accuracy: 0.9978 | Precision: 0.9017\n",
            "\n",
            "🔄 Epoch 268/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 268/350 | Loss: 0.0079 | Accuracy: 0.9976 | Precision: 0.8901\n",
            "\n",
            "🔄 Epoch 269/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 269/350 | Loss: 0.0079 | Accuracy: 0.9976 | Precision: 0.8952\n",
            "\n",
            "🔄 Epoch 270/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 270/350 | Loss: 0.0080 | Accuracy: 0.9975 | Precision: 0.8938\n",
            "\n",
            "🔄 Epoch 271/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 271/350 | Loss: 0.0081 | Accuracy: 0.9975 | Precision: 0.8914\n",
            "\n",
            "🔄 Epoch 272/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 272/350 | Loss: 0.0080 | Accuracy: 0.9976 | Precision: 0.8943\n",
            "\n",
            "🔄 Epoch 273/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 273/350 | Loss: 0.0083 | Accuracy: 0.9975 | Precision: 0.8906\n",
            "\n",
            "🔄 Epoch 274/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 274/350 | Loss: 0.0086 | Accuracy: 0.9974 | Precision: 0.8885\n",
            "\n",
            "🔄 Epoch 275/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 275/350 | Loss: 0.0082 | Accuracy: 0.9975 | Precision: 0.8921\n",
            "\n",
            "🔄 Epoch 276/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 276/350 | Loss: 0.0079 | Accuracy: 0.9976 | Precision: 0.8951\n",
            "\n",
            "🔄 Epoch 277/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 277/350 | Loss: 0.0080 | Accuracy: 0.9975 | Precision: 0.8973\n",
            "\n",
            "🔄 Epoch 278/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 278/350 | Loss: 0.0082 | Accuracy: 0.9975 | Precision: 0.8910\n",
            "\n",
            "🔄 Epoch 279/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 279/350 | Loss: 0.0085 | Accuracy: 0.9974 | Precision: 0.8889\n",
            "\n",
            "🔄 Epoch 280/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 280/350 | Loss: 0.0084 | Accuracy: 0.9974 | Precision: 0.8934\n",
            "\n",
            "🔄 Epoch 281/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 281/350 | Loss: 0.0079 | Accuracy: 0.9975 | Precision: 0.8953\n",
            "\n",
            "🔄 Epoch 282/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 282/350 | Loss: 0.0076 | Accuracy: 0.9977 | Precision: 0.9005\n",
            "\n",
            "🔄 Epoch 283/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 283/350 | Loss: 0.0077 | Accuracy: 0.9976 | Precision: 0.8995\n",
            "\n",
            "🔄 Epoch 284/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 284/350 | Loss: 0.0075 | Accuracy: 0.9977 | Precision: 0.8984\n",
            "\n",
            "🔄 Epoch 285/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 285/350 | Loss: 0.0077 | Accuracy: 0.9976 | Precision: 0.8986\n",
            "\n",
            "🔄 Epoch 286/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 286/350 | Loss: 0.0075 | Accuracy: 0.9977 | Precision: 0.9026\n",
            "\n",
            "🔄 Epoch 287/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 287/350 | Loss: 0.0078 | Accuracy: 0.9976 | Precision: 0.8968\n",
            "\n",
            "🔄 Epoch 288/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 288/350 | Loss: 0.0087 | Accuracy: 0.9974 | Precision: 0.8907\n",
            "\n",
            "🔄 Epoch 289/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 289/350 | Loss: 0.0078 | Accuracy: 0.9976 | Precision: 0.8964\n",
            "\n",
            "🔄 Epoch 290/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 290/350 | Loss: 0.0092 | Accuracy: 0.9972 | Precision: 0.8843\n",
            "\n",
            "🔄 Epoch 291/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 291/350 | Loss: 0.0086 | Accuracy: 0.9974 | Precision: 0.8941\n",
            "\n",
            "🔄 Epoch 292/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 292/350 | Loss: 0.0076 | Accuracy: 0.9977 | Precision: 0.9013\n",
            "\n",
            "🔄 Epoch 293/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 293/350 | Loss: 0.0072 | Accuracy: 0.9978 | Precision: 0.9061\n",
            "\n",
            "🔄 Epoch 294/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 294/350 | Loss: 0.0068 | Accuracy: 0.9979 | Precision: 0.9097\n",
            "\n",
            "🔄 Epoch 295/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 295/350 | Loss: 0.0069 | Accuracy: 0.9979 | Precision: 0.9099\n",
            "\n",
            "🔄 Epoch 296/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 296/350 | Loss: 0.0068 | Accuracy: 0.9979 | Precision: 0.9079\n",
            "\n",
            "🔄 Epoch 297/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 297/350 | Loss: 0.0069 | Accuracy: 0.9979 | Precision: 0.9060\n",
            "\n",
            "🔄 Epoch 298/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 298/350 | Loss: 0.0072 | Accuracy: 0.9978 | Precision: 0.9046\n",
            "\n",
            "🔄 Epoch 299/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 299/350 | Loss: 0.0078 | Accuracy: 0.9976 | Precision: 0.8960\n",
            "\n",
            "🔄 Epoch 300/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 300/350 | Loss: 0.0074 | Accuracy: 0.9978 | Precision: 0.9022\n",
            "\n",
            "🔄 Epoch 301/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 301/350 | Loss: 0.0076 | Accuracy: 0.9977 | Precision: 0.9040\n",
            "\n",
            "🔄 Epoch 302/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 302/350 | Loss: 0.0070 | Accuracy: 0.9979 | Precision: 0.9064\n",
            "\n",
            "🔄 Epoch 303/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 303/350 | Loss: 0.0066 | Accuracy: 0.9980 | Precision: 0.9110\n",
            "\n",
            "🔄 Epoch 304/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 304/350 | Loss: 0.0068 | Accuracy: 0.9979 | Precision: 0.9103\n",
            "\n",
            "🔄 Epoch 305/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 305/350 | Loss: 0.0069 | Accuracy: 0.9979 | Precision: 0.9062\n",
            "\n",
            "🔄 Epoch 306/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 306/350 | Loss: 0.0071 | Accuracy: 0.9978 | Precision: 0.9046\n",
            "\n",
            "🔄 Epoch 307/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 307/350 | Loss: 0.0071 | Accuracy: 0.9979 | Precision: 0.9071\n",
            "\n",
            "🔄 Epoch 308/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 308/350 | Loss: 0.0071 | Accuracy: 0.9979 | Precision: 0.9074\n",
            "\n",
            "🔄 Epoch 309/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 309/350 | Loss: 0.0071 | Accuracy: 0.9978 | Precision: 0.9028\n",
            "\n",
            "🔄 Epoch 310/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 310/350 | Loss: 0.0075 | Accuracy: 0.9977 | Precision: 0.9021\n",
            "\n",
            "🔄 Epoch 311/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 311/350 | Loss: 0.0116 | Accuracy: 0.9971 | Precision: 0.8780\n",
            "\n",
            "🔄 Epoch 312/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 312/350 | Loss: 0.0136 | Accuracy: 0.9960 | Precision: 0.8375\n",
            "\n",
            "🔄 Epoch 313/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 313/350 | Loss: 0.0090 | Accuracy: 0.9972 | Precision: 0.8780\n",
            "\n",
            "🔄 Epoch 314/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 314/350 | Loss: 0.0074 | Accuracy: 0.9977 | Precision: 0.9012\n",
            "\n",
            "🔄 Epoch 315/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 315/350 | Loss: 0.0061 | Accuracy: 0.9981 | Precision: 0.9118\n",
            "\n",
            "🔄 Epoch 316/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 316/350 | Loss: 0.0064 | Accuracy: 0.9980 | Precision: 0.9098\n",
            "\n",
            "🔄 Epoch 317/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 317/350 | Loss: 0.0057 | Accuracy: 0.9982 | Precision: 0.9195\n",
            "\n",
            "🔄 Epoch 318/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 318/350 | Loss: 0.0057 | Accuracy: 0.9983 | Precision: 0.9171\n",
            "\n",
            "🔄 Epoch 319/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 319/350 | Loss: 0.0057 | Accuracy: 0.9983 | Precision: 0.9203\n",
            "\n",
            "🔄 Epoch 320/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 320/350 | Loss: 0.0057 | Accuracy: 0.9983 | Precision: 0.9226\n",
            "\n",
            "🔄 Epoch 321/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 321/350 | Loss: 0.0061 | Accuracy: 0.9982 | Precision: 0.9171\n",
            "\n",
            "🔄 Epoch 322/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 322/350 | Loss: 0.0062 | Accuracy: 0.9981 | Precision: 0.9175\n",
            "\n",
            "🔄 Epoch 323/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 323/350 | Loss: 0.0064 | Accuracy: 0.9981 | Precision: 0.9117\n",
            "\n",
            "🔄 Epoch 324/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 324/350 | Loss: 0.0065 | Accuracy: 0.9980 | Precision: 0.9123\n",
            "\n",
            "🔄 Epoch 325/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 325/350 | Loss: 0.0065 | Accuracy: 0.9980 | Precision: 0.9128\n",
            "\n",
            "🔄 Epoch 326/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 326/350 | Loss: 0.0065 | Accuracy: 0.9980 | Precision: 0.9157\n",
            "\n",
            "🔄 Epoch 327/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 327/350 | Loss: 0.0068 | Accuracy: 0.9980 | Precision: 0.9100\n",
            "\n",
            "🔄 Epoch 328/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 328/350 | Loss: 0.0065 | Accuracy: 0.9980 | Precision: 0.9143\n",
            "\n",
            "🔄 Epoch 329/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 329/350 | Loss: 0.0066 | Accuracy: 0.9980 | Precision: 0.9129\n",
            "\n",
            "🔄 Epoch 330/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 330/350 | Loss: 0.0100 | Accuracy: 0.9972 | Precision: 0.8763\n",
            "\n",
            "🔄 Epoch 331/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 331/350 | Loss: 0.0083 | Accuracy: 0.9976 | Precision: 0.8916\n",
            "\n",
            "🔄 Epoch 332/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 332/350 | Loss: 0.0074 | Accuracy: 0.9978 | Precision: 0.9045\n",
            "\n",
            "🔄 Epoch 333/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 333/350 | Loss: 0.0058 | Accuracy: 0.9982 | Precision: 0.9176\n",
            "\n",
            "🔄 Epoch 334/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 334/350 | Loss: 0.0053 | Accuracy: 0.9984 | Precision: 0.9235\n",
            "\n",
            "🔄 Epoch 335/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 335/350 | Loss: 0.0051 | Accuracy: 0.9984 | Precision: 0.9258\n",
            "\n",
            "🔄 Epoch 336/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 336/350 | Loss: 0.0052 | Accuracy: 0.9984 | Precision: 0.9234\n",
            "\n",
            "🔄 Epoch 337/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 337/350 | Loss: 0.0060 | Accuracy: 0.9982 | Precision: 0.9163\n",
            "\n",
            "🔄 Epoch 338/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 338/350 | Loss: 0.0087 | Accuracy: 0.9977 | Precision: 0.8995\n",
            "\n",
            "🔄 Epoch 339/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 339/350 | Loss: 0.0125 | Accuracy: 0.9965 | Precision: 0.8586\n",
            "\n",
            "🔄 Epoch 340/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 340/350 | Loss: 0.0075 | Accuracy: 0.9977 | Precision: 0.9002\n",
            "\n",
            "🔄 Epoch 341/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 341/350 | Loss: 0.0059 | Accuracy: 0.9982 | Precision: 0.9179\n",
            "\n",
            "🔄 Epoch 342/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 342/350 | Loss: 0.0055 | Accuracy: 0.9983 | Precision: 0.9257\n",
            "\n",
            "🔄 Epoch 343/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 343/350 | Loss: 0.0051 | Accuracy: 0.9984 | Precision: 0.9303\n",
            "\n",
            "🔄 Epoch 344/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 344/350 | Loss: 0.0049 | Accuracy: 0.9985 | Precision: 0.9321\n",
            "\n",
            "🔄 Epoch 345/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 345/350 | Loss: 0.0057 | Accuracy: 0.9983 | Precision: 0.9222\n",
            "\n",
            "🔄 Epoch 346/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 346/350 | Loss: 0.0062 | Accuracy: 0.9982 | Precision: 0.9153\n",
            "\n",
            "🔄 Epoch 347/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 347/350 | Loss: 0.0061 | Accuracy: 0.9982 | Precision: 0.9140\n",
            "\n",
            "🔄 Epoch 348/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 348/350 | Loss: 0.0064 | Accuracy: 0.9981 | Precision: 0.9099\n",
            "\n",
            "🔄 Epoch 349/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": []
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 349/350 | Loss: 0.0060 | Accuracy: 0.9982 | Precision: 0.9190\n",
            "\n",
            "🔄 Epoch 350/350\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "                                                                "
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Epoch 350/350 | Loss: 0.0058 | Accuracy: 0.9983 | Precision: 0.9193\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Save model to Google Drive or local\n",
        "torch.save(model.state_dict(), '/content/drive/My Drive/CrackDetection/crack_segmentation_model.pth')\n",
        "print(\"Model saved!\")"
      ],
      "metadata": {
        "id": "XHPH_evx18j3",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "bfc71ad1-9f96-4005-9bb8-01417f329efd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model saved!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "---\n",
        "#**Visualizing Model Performance Over Epochs**\n",
        "\n",
        "---\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "z3PtNSupPhug"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "epochs_range = range(1, EPOCHS + 1)\n",
        "\n",
        "plt.figure(figsize=(18, 5))\n",
        "\n",
        "# 📉 Loss\n",
        "plt.subplot(1, 3, 1)\n",
        "plt.plot(epochs_range, history[\"loss\"], color='purple', label=\"Loss\")\n",
        "plt.title(\"Training Loss\")\n",
        "plt.xlabel(\"Epoch\")\n",
        "plt.ylabel(\"Loss\")\n",
        "plt.grid(True)\n",
        "plt.legend()\n",
        "\n",
        "#  Accuracy\n",
        "plt.subplot(1, 3, 2)\n",
        "plt.plot(epochs_range, history[\"accuracy\"], color='green', label=\"Accuracy\")\n",
        "plt.title(\"Training Accuracy\")\n",
        "plt.xlabel(\"Epoch\")\n",
        "plt.ylabel(\"Accuracy\")\n",
        "plt.grid(True)\n",
        "plt.legend()\n",
        "\n",
        "#  Precision\n",
        "plt.subplot(1, 3, 3)\n",
        "plt.plot(epochs_range, history[\"precision\"], color='blue', label=\"Precision\")\n",
        "plt.title(\"Training Precision\")\n",
        "plt.xlabel(\"Epoch\")\n",
        "plt.ylabel(\"Precision\")\n",
        "plt.grid(True)\n",
        "plt.legend()\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "plt.savefig(\"training_graphs.png\")\n"
      ],
      "metadata": {
        "id": "E-w5G8Itgz_J",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 421
        },
        "outputId": "0d18701f-3a70-4ad8-ce3e-3486a2eed95a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1800x500 with 3 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAABv4AAAHqCAYAAADMEzkrAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQABAABJREFUeJzs3Xd4VNXWx/HvpCeQQIDQe5CO9I4U6SIKoiAWioJYsMVrwSsiNq5XQfCK8ooiIiIKgoUeEUSKICAovXcSOiF9kpn3j+3MZEgCgUwqv8/z5Dll9tlnzwb0ZNastS12u92OiIiIiIiIiIiIiIiIiBRoXnk9ABERERERERERERERERHJPgX+RERERERERERERERERAoBBf5ERERERERERERERERECgEF/kREREREREREREREREQKAQX+RERERERERERERERERAoBBf5ERERERERERERERERECgEF/kREREREREREREREREQKAQX+RERERERERERERERERAoBBf5ERERERERERERERERECgEF/kQkXxoyZAhVq1a9rmtfe+01LBaLZwckIiIikkv0HCQiIiLiOXq28oyqVasyZMiQa7pG8yeSNxT4E5FrYrFYsvSzcuXKvB5qnhgyZAhFixbN62GIiIhIDtBzUNb1798fi8XCiy++mNdDERERkXxKz1ZXNmTIELd5CAkJoWHDhowfP56kpKS8Hp6I5GMWu91uz+tBiEjBMXPmTLfjGTNmEBkZyZdfful2vmvXrpQpU+a672O1WrHZbPj7+1/ztSkpKaSkpBAQEHDd979eQ4YMYe7cucTGxub6vUVERCRn6Tkoa2JiYihTpgxly5YlNTWVw4cP65veIiIiko6era5syJAhzJ49m08//RSACxcu8N1337Fy5UoGDBjA7Nmzc3U8SUlJeHl54evrm+Vr8sOzqciNSIE/EcmWkSNHMnnyZK72n5L4+HiCgoJyaVR5R4E/ERGRG4eegzL2+eefM2LECJYuXcqtt97KypUr6dChQ14PKx273U5iYiKBgYF5PRQRERFBz1aXy+gzJpvNRsuWLdm4cSPHjx+nfPny6a7TM46IqNSniHhcx44dqV+/Pps2baJ9+/YEBQXx8ssvA/DDDz/Qq1cvypcvj7+/P+Hh4bzxxhukpqa69XF5/fVDhw5hsVh47733+OSTTwgPD8ff35/mzZvzxx9/uF2bUf1wi8XCyJEj+f7776lfvz7+/v7Uq1ePJUuWpBv/ypUradasGQEBAYSHh/N///d/Hq9JPmfOHJo2bUpgYCClSpXigQce4Pjx425toqKiGDp0KBUrVsTf359y5cpx5513cujQIWebjRs30r17d0qVKkVgYCDVqlXjoYce8tg4RURE5NroOQi++uorunbtSqdOnahTpw5fffVVhu127dpF//79CQsLIzAwkFq1avHvf//brc3x48d5+OGHnXNWrVo1HnvsMZKTkzN9vwDTp0/HYrG4PTdVrVqV22+/naVLl9KsWTMCAwP5v//7P8AEK2+99VZKly6Nv78/devW5eOPP85w3IsXL6ZDhw4EBwcTEhJC8+bNmTVrFgBjxozB19eX06dPp7vukUceoXjx4iQmJl59EkVERATQs9XlvLy86Nixo/N9wJWfcS5cuMAzzzxDpUqV8Pf3p0aNGrzzzjvYbDa3fm02G5MmTaJBgwYEBAQQFhZGjx492Lhxo7PN5Wv8Wa1Wxo4dy0033URAQAAlS5akXbt2REZGXnH+UlJSeOONN5zzXrVqVV5++eV05Usd72v16tW0aNGCgIAAqlevzowZM65r7kRuJD55PQARKZzOnj1Lz549uffee3nggQecJRmmT59O0aJFiYiIoGjRovzyyy+8+uqrxMTE8O67716131mzZnHp0iVGjBiBxWLhv//9L3fddRcHDhy4aqmB1atXM2/ePB5//HGCg4P54IMP6NevH0eOHKFkyZIA/Pnnn/To0YNy5coxduxYUlNTef311wkLC8v+pPxj+vTpDB06lObNmzNu3Diio6OZNGkSa9as4c8//6R48eIA9OvXj+3bt/Pkk09StWpVTp06RWRkJEeOHHEed+vWjbCwMF566SWKFy/OoUOHmDdvnsfGKiIiItfuRn4OOnHiBCtWrOCLL74AYODAgbz//vt8+OGH+Pn5Odv99ddf3HLLLfj6+vLII49QtWpV9u/fz08//cRbb73l7KtFixZcuHCBRx55hNq1a3P8+HHmzp1LfHy8W39ZtXv3bgYOHMiIESMYPnw4tWrVAuDjjz+mXr163HHHHfj4+PDTTz/x+OOPY7PZeOKJJ5zXT58+nYceeoh69eoxatQoihcvzp9//smSJUu47777ePDBB3n99df55ptvGDlypPO65ORk5s6dS79+/VTqSkRE5BrdyM9WGdm/fz+A8z6Q8TNOfHw8HTp04Pjx44wYMYLKlSuzdu1aRo0axcmTJ5k4caLz+ocffpjp06fTs2dPhg0bRkpKCr/99hu///47zZo1y3Acr732GuPGjWPYsGG0aNGCmJgYNm7cyObNm+natWum4x82bBhffPEFd999N8899xzr169n3Lhx7Ny5k/nz57u13bdvH3fffTcPP/wwgwcPZtq0aQwZMoSmTZtSr169bMyiSCFnFxHJhieeeMJ++X9KOnToYAfsU6ZMSdc+Pj4+3bkRI0bYg4KC7ImJic5zgwcPtlepUsV5fPDgQTtgL1mypP3cuXPO8z/88IMdsP/000/Oc2PGjEk3JsDu5+dn37dvn/Pc1q1b7YD9f//7n/Nc79697UFBQfbjx487z+3du9fu4+OTrs+MDB482F6kSJFMX09OTraXLl3aXr9+fXtCQoLz/IIFC+yA/dVXX7Xb7Xb7+fPn7YD93XffzbSv+fPn2wH7H3/8cdVxiYiIiOfpOSi99957zx4YGGiPiYmx2+12+549e+yAff78+W7t2rdvbw8ODrYfPnzY7bzNZnPuDxo0yO7l5ZXhs46jXUbv12632z///HM7YD948KDzXJUqVeyAfcmSJenaZ/Rn0717d3v16tWdxxcuXLAHBwfbW7Zs6fYcd/m4W7dubW/ZsqXb6/PmzbMD9hUrVqS7j4iIiBh6tnLn+Izp9OnT9tOnT9v37dtnf/vtt+0Wi8V+8803O9tl9ozzxhtv2IsUKWLfs2eP2/mXXnrJ7u3tbT9y5Ijdbrfbf/nlFztgf+qpp9KNIe0zTpUqVeyDBw92Hjds2NDeq1evK76Hy+dvy5YtdsA+bNgwt3b/+te/7ID9l19+Sfe+Vq1a5Tx36tQpu7+/v/2555674n1FbnQq9SkiOcLf35+hQ4emO5+2vvilS5c4c+YMt9xyC/Hx8ezateuq/Q4YMIDQ0FDn8S233ALAgQMHrnptly5dCA8Pdx7ffPPNhISEOK9NTU3l559/pk+fPm410mvUqEHPnj2v2n9WbNy4kVOnTvH444+7fdu7V69e1K5dm4ULFwJmnvz8/Fi5ciXnz5/PsC9HZuCCBQuwWq0eGZ+IiIhk3438HPTVV1/Rq1cvgoODAbjpppto2rSpW7nP06dPs2rVKh566CEqV67sdr2jFJTNZuP777+nd+/eGX7L/HrLY1WrVo3u3bunO5/2z+bixYucOXOGDh06cODAAS5evAhAZGQkly5d4qWXXkqXtZd2PIMGDWL9+vXOb+ODmZdKlSrly7UORURE8rsb+dkqLi6OsLAwwsLCqFGjBi+//DKtW7dOlxmX0TPOnDlzuOWWWwgNDeXMmTPOny5dupCamsqqVasA+O6777BYLIwZMybd/a/0zFW8eHG2b9/O3r17s/x+Fi1aBEBERITb+eeeew7A+bmYQ926dZ1/LgBhYWHUqlUrS39GIjcyBf5EJEdUqFAhw/JL27dvp2/fvhQrVoyQkBDCwsJ44IEHAJwfqlzJ5R8OOR7QMguOXelax/WOa0+dOkVCQgI1atRI1y6jc9fj8OHDAM6yUmnVrl3b+bq/vz/vvPMOixcvpkyZMrRv357//ve/REVFOdt36NCBfv36MXbsWEqVKsWdd97J559/nq4muoiIiOSuG/U5aOfOnfz555+0bduWffv2OX86duzIggULiImJAVwfptWvXz/Tvk6fPk1MTMwV21yPatWqZXh+zZo1dOnShSJFilC8eHHCwsKc6wc5/mwcgbyrjWnAgAH4+/s7g50XL15kwYIF3H///R5dM1pERORGcaM+WwEEBAQQGRlJZGQkq1at4ujRo6xZs4bq1au7tcvoGWfv3r0sWbLEGTh0/HTp0sU5RjDPOOXLl6dEiRJZHhfA66+/zoULF6hZsyYNGjTg+eef56+//rriNYcPH8bLyyvdHJQtW5bixYs7PxdzuNo8i0jGtMafiOSItN+6crhw4QIdOnQgJCSE119/nfDwcAICAti8eTMvvvhiuoWFM+Lt7Z3hebvdnqPX5oVnnnmG3r178/3337N06VJGjx7NuHHj+OWXX2jcuDEWi4W5c+fy+++/89NPP7F06VIeeughxo8fz++//07RokXz+i2IiIjckG7U56CZM2cC8Oyzz/Lss8+me/27777L8Nv62ZFZIC01NTXD8xn92ezfv5/OnTtTu3ZtJkyYQKVKlfDz82PRokW8//77WfqzSSs0NJTbb7+dr776ildffZW5c+eSlJTk/CBSRERErs2N+mzluI8jUHclGc2RzWaja9euvPDCCxleU7NmzWyNrX379uzfv58ffviBZcuW8emnn/L+++8zZcoUhg0bdsVrs/plqIL2WZ5IfqHAn4jkmpUrV3L27FnmzZtH+/btnecPHjyYh6NyKV26NAEBAezbty/daxmdux5VqlQBzKLLt956q9tru3fvdr7uEB4eznPPPcdzzz3H3r17adSoEePHj3d+sAbQqlUrWrVqxVtvvcWsWbO4//77mT179lUfskRERCT3FPbnILvdzqxZs+jUqROPP/54utffeOMNvvrqK4YOHer8hvq2bdsy7S8sLIyQkJArtgHXN/MvXLjgLIMOpPu2+JX89NNPJCUl8eOPP7p9q3zFihVu7RzlvLZt23bVb+oPGjSIO++8kz/++IOvvvqKxo0bU69evSyPSURERK6ssD9beUJ4eDixsbFXDRyGh4ezdOlSzp07d81ZfyVKlGDo0KEMHTqU2NhY2rdvz2uvvZbpZ1JVqlTBZrOxd+9e6tSp4zwfHR3NhQsX0n0uJiLXR6U+RSTXOL6lk/ZbOcnJyXz00Ud5NSQ3jm9Rff/995w4ccJ5ft++fSxevNgj92jWrBmlS5dmypQpbiU5Fy9ezM6dO+nVqxcA8fHxJCYmul0bHh5OcHCw87rz58+n+4ZTo0aNAFTuU0REJJ8p7M9Ba9as4dChQwwdOpS777473c+AAQNYsWIFJ06cICwsjPbt2zNt2jSOHDni1o9jfry8vOjTpw8//fQTGzduTHc/RztHMM6xRg2YtXC++OKLa3rvafsEUx7s888/d2vXrVs3goODGTduXLrntMufyXr27EmpUqV45513+PXXX5XtJyIi4mGF/dnKE/r378+6detYunRputcuXLhASkoKAP369cNutzN27Nh07a6UWXf27Fm346JFi1KjRo0rfiZ12223ATBx4kS38xMmTABwfi4mItmjjD8RyTVt2rQhNDSUwYMH89RTT2GxWPjyyy/zVXr+a6+9xrJly2jbti2PPfYYqampfPjhh9SvX58tW7ZkqQ+r1cqbb76Z7nyJEiV4/PHHeeeddxg6dCgdOnRg4MCBREdHM2nSJKpWreosi7Vnzx46d+5M//79qVu3Lj4+PsyfP5/o6GjuvfdeAL744gs++ugj+vbtS3h4OJcuXWLq1KmEhIQ4H6REREQkfyjsz0FfffUV3t7emX5Yc8cdd/Dvf/+b2bNnExERwQcffEC7du1o0qQJjzzyCNWqVePQoUMsXLjQea+3336bZcuW0aFDBx555BHq1KnDyZMnmTNnDqtXr6Z48eJ069aNypUr8/DDD/P888/j7e3NtGnTCAsLSxdUzEy3bt3w8/Ojd+/ejBgxgtjYWKZOnUrp0qU5efKks11ISAjvv/8+w4YNo3nz5tx3332EhoaydetW4uPj3YKNvr6+3HvvvXz44Yd4e3szcODALI1FREREsqawP1t5wvPPP8+PP/7I7bffzpAhQ2jatClxcXH8/fffzJ07l0OHDlGqVCk6derEgw8+yAcffMDevXvp0aMHNpuN3377jU6dOjFy5MgM+69bty4dO3akadOmlChRgo0bNzJ37txM2wM0bNiQwYMH88knnzjLtW7YsIEvvviCPn360KlTp5yaDpEbigJ/IpJrSpYsyYIFC3juued45ZVXCA0N5YEHHqBz58507949r4cHQNOmTVm8eDH/+te/GD16NJUqVeL1119n586d7Nq1K0t9JCcnM3r06HTnw8PDefzxxxkyZAhBQUH85z//4cUXX6RIkSL07duXd955x1miqlKlSgwcOJDly5fz5Zdf4uPjQ+3atfn222/p168fgPPhaPbs2URHR1OsWDFatGjBV199leGiziIiIpJ3CvNzkNVqZc6cObRp0ybT8lD169enWrVqzJw5k4iICBo2bMjvv//O6NGj+fjjj0lMTKRKlSr079/feU2FChVYv349o0eP5quvviImJoYKFSrQs2dPgoKCABNgmz9/Po8//jijR4+mbNmyPPPMM4SGhmZ5PcFatWoxd+5cXnnlFf71r39RtmxZHnvsMcLCwnjooYfc2j788MOULl2a//znP7zxxhv4+vpSu3btDNc0HDRoEB9++CGdO3emXLlyWRqLiIiIZE1hfrbylKCgIH799Vfefvtt5syZw4wZMwgJCaFmzZqMHTuWYsWKOdt+/vnn3HzzzXz22Wc8//zzFCtWjGbNmtGmTZtM+3/qqaf48ccfWbZsGUlJSVSpUoU333yT559//orj+vTTT6levTrTp09n/vz5lC1bllGjRjFmzBiPvXeRG53Fnp++BiEikk/16dOH7du3s3fv3rweioiIiEiu0nPQ9dm6dSuNGjVixowZPPjgg3k9HBEREckn9GwlIjlNa/yJiFwmISHB7Xjv3r0sWrSIjh075s2ARERERHKJnoM8Z+rUqRQtWpS77rorr4ciIiIieUTPViKSF1TqU0TkMtWrV2fIkCFUr16dw4cP8/HHH+Pn58cLL7yQ10MTERERyVF6Dsq+n376iR07dvDJJ58wcuRIihQpktdDEhERkTyiZysRyQsq9SkicpmhQ4eyYsUKoqKi8Pf3p3Xr1rz99ts0adIkr4cmIiIikqP0HJR9VatWJTo6mu7du/Pll18SHByc10MSERGRPKJnKxHJCwr8iYiIiIiIiIiIiIiIiBQCWuNPREREREREREREREREpBBQ4E9ERERERERERERERESkEPDJ6wHkNpvNxokTJwgODsZiseT1cERERCSfsdvtXLp0ifLly+Plpe9IXYmeq0RERORK9FyVdXquEhERkSu5lueqGy7wd+LECSpVqpTXwxAREZF87ujRo1SsWDGvh5Gv6blKREREskLPVVen5yoRERHJiqw8V91wgb/g4GDATE5ISIhH+7ZarSxbtoxu3brh6+vr0b4LIs2HO82HO82HO82HO82HO82Hu5yej5iYGCpVquR8ZpDM6bkq92g+3Gk+3Gk+3Gk+3Gk+3Gk+3Om5Kv/Qc1Xu0Xy403y403y403y403y403y4y0/PVTdc4M9RLiEkJCRHHqSCgoIICQnRX3Q0H5fTfLjTfLjTfLjTfLjTfLjLrflQiaWr03NV7tF8uNN8uNN8uNN8uNN8uNN8uNNzVf6h56rco/lwp/lwp/lwp/lwp/lwp/lwl5+eq1RgXURERERERERERERERKQQUOBPREREREREREREREREpBBQ4E9ERERERERERERERESkELjh1vgTEREpbFJTU7FarXk9jFxjtVrx8fEhMTGR1NTUa77e19cXb2/vHBiZiIiIiEjOup5n/+w+Pxc2uTkf+t1DRETyggJ/IiIiBZTdbicqKooLFy7k9VByld1up2zZshw9ejRLCxpnpHjx4pQtW/a6rxcRERERyU3Zefb3xPNzYZLb86HfPUREJLcp8CciIlJAOX7xL126NEFBQTfML5I2m43Y2FiKFi2Kl9e1VS232+3Ex8dz6tQpAMqVK5cTQxQRERER8ajsPPtn5/m5MMqt+dDvHiIiklcU+BMRESmAUlNTnb/4lyxZMq+Hk6tsNhvJyckEBARc1y/qgYGBAJw6dYrSpUur9I6IiIiI5GvZffbP7vNzYZOb86HfPUREJC/o//YiIiIFkGNdj6CgoDweScHkmLcbaW1EERERESmY9OxfsOl3DxERyW0K/ImI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FXVS55o6daqefvpp5eXlKTU1VS+88IL69+9/zP0PHDig+++/X++995727dunjh07asqUKRoyZEid3lO48nikl14yb3v8cenmm+t23uoKfwAAwKxWhb9t27bJYrGoQ4cOkqRly5Zp9uzZ6tmzJ9NZ6XDHn9ug8AcAQBN25513qlOnTsrJyVGnTp20bNky7d27V3fffbf+8Y9/hDq8sOAvljLVJwAAqKHo6GiNGTOmTueYO3eusrKyNG3aNKWlpWnKlCkaNGiQ1q1bp4SEhCr7V1RUaMCAAUpISNA777yj9u3ba+vWrWrRokWd4ghnn30mbdwoNW/uvb99u3TllXU/r6/Q55vqMzq67ucEAKCxqVXh79prr9Utt9yi66+/Xnl5eRowYIBOP/10vfnmm8rLy9P48eMDHWdY8X2T33exCgAAND1LlizRwoULFR8fL6vVKqvVqvPPP1+TJk3SHXfcoe+//z7UITZ4TPUJAABO5KOPPtJll10mu92ujz766Lj71nSq9cmTJ2vMmDEaPXq0JGnatGn65JNPNHPmTN17771V9p85c6b27dunr7/+Wna7XZKUkpJycm+kkXn/fe/t9ddLaWnen0Cg4w8AgBOrVeHvxx9/9E9v8NZbb6lXr1766quvtGDBAt16660U/qyHC390/AEA0GS53W41a9ZMkhQfH6+dO3eqe/fu6tixo9atWxfi6MKDv/BHxx8AADiGYcOGKS8vTwkJCRo2bNgx97NYLDWaar2iokLLly/XuHHj/NusVqsyMjK0ZMmSao/56KOPlJ6erttvv10ffvih2rRpo2uvvVZ///vfZavtYnZhbtUq7+355wf2vBT+AAA4sVoV/lwul5xOpyTp888/939jqkePHtq1a1fgogtXvmtT1P0AAGiyevXqpR9++EGdOnVSWlqannrqKTkcDv3zn/9U586dQx1eWKHjDwAAHIvH46n2fm0VFBTI7XYrMTHRtD0xMdG/buCvbdq0SQsXLtTIkSM1b948bdiwQbfddptcLpcmTJhQ7THl5eUqLy/3Py4sLJTkvebmcrnq/D6O5jtfoM97LB6PtHp1hCSLevZ0KZAvGxFhkRShsjLv46goj1yuk1s7u77z0dCRDzPyYUY+zMiHGfkwC3Y+Tua8tSr8nX766Zo2bZouv/xyZWdn65FHHpEk7dy5U61bt67NKRsVOv4A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OhqzFi3MtwAAwIxRUBCwxh8AABg6dKg++OADPf7443rnnXcUFRWl1NRULVy4UK1atQp1eGGFjj8AAFAT3377rXJycrRgwQL17t1bMTExpuffe++9EEUWfr7+2nubnh66KT2P5aqrpI0bpWuuCXUkAAA0TBT+goA1/gAAgCRdfvnluvzyyyVJhYWF+ve//62//vWvWr58udxud4ija/j8U302tKtNAACgQWrRooWuuuqqUIfRKPgKf+eeG9o4qtO8ufTYY6GOAgCAhovCXzCwxh8AADjsyy+/1IwZM/Tuu++qXbt2uvLKKzV16tRQhxUWWOMPAADUhMfj0dNPP63169eroqJCl1xyiR566CFFRUWFOrSw5HJJ//uf9356emhjAQAAJ4/CXxCwxh8AAE1bXl6eZs2apRkzZqiwsFB//OMfVV5erg8++EA9e/YMdXhhh44/AABwPI899pgeeughZWRkKCoqSs8//7z27NmjmTNnhjq0sPTpp9Levd71/dLSQh0NAAA4WdZQB9Ao0fEHAECTNXToUHXv3l2rVq3SlClTtHPnTr3wwguhDiss0fEHAABq4rXXXtNLL72kzz77TB988IE+/vhjvfnmm/J4PKEOLSy99pr3duRIKYKWAQAAwg6/voOANf4AAGi6Pv30U91xxx0aO3asunXrFupwwppv9gSrhe+qAQCAY8vNzdWQIUP8jzMyMmSxWLRz50516NAhhJGFn337pI8/9t4fNSq0sQAAgNrhKkoQ+KajouMPAICmZ/HixTp06JD69euntLQ0vfjiiyooKAh1WGHJ3/HHVJ8AAOA4KisrFRkZadpmt9vlcrlCFFH4eu89qaJCSk2V+vQJdTQAAKA26PgLAtb4AwCg6TrnnHN0zjnnaMqUKZo7d65mzpyprKwseTweZWdnKzk5Wc2aNQt1mGGBqT4BAEBNGIahG2+8UU6n07+trKxMt956q2JiYvzb3nvvvVCEF1beest7O2JEaOMAAAC1R8dfMLDGHwAATV5MTIxuuukmLV68WKtXr9bdd9+tJ554QgkJCfrd734X6vDCAh1/AACgJm644QYlJCSoefPm/p/rrrtO7dq1M23D8RUUSAsXeu//4Q+hjQUAANQeHX9B4O/4o/AHAAAkde/eXU899ZQmTZqkjz/+WDNnzgx1SOHh8FCKjj8AAHA8r7zySqhDaBQ++EByu6W+faWuXUMdDQAAqC06/oLA/6106n4AAOAoNptNw4YN00cffRTqUMICHX8AAAD1w+2WnnvOe/+PfwxtLAAAoG4o/AUBHX8AAAB1xxp/AAAA9eP116Uff5RatJD+3/8LdTQAAKAuKPwFg6/hz6DwBwAAAmfq1KlKSUlRZGSk0tLStGzZsuPuP2XKFHXv3l1RUVFKTk7WX/7yF5WVlfmff+ihh2SxWEw/PXr0CPbbOGl0/AEAAASPYUgPPeS9f999UqtWIQ0HAADUEWv8BYHv4hQdfwAAIFDmzp2rrKwsTZs2TWlpaZoyZYoGDRqkdevWKSEhocr+s2fP1r333quZM2fq3HPP1fr163XjjTfKYrFo8uTJ/v1OP/10ff755/7HERENZ3jo6/izWviuGgAAQLBs2SJt3So5HFJmZqijAQAAdcVVlCDwTfXJGn8AACBQJk+erDFjxmj06NHq2bOnpk2bpujoaM2cObPa/b/++mudd955uvbaa5WSkqKBAwfqmmuuqdIlGBERoaSkJP9PfHx8fbydGvHNnsBUnwAAAMGzfLn3tndvKSoqtLEAAIC6o/AXBHT8AQCAQKqoqNDy5cuVkZHh32a1WpWRkaElS5ZUe8y5556r5cuX+wt9mzZt0rx58zRkyBDTfr/88ovatWunzp07a+TIkcrNzQ3eGzlJ/jX+mOoTAAAgaFas8N6eeWZo4wAAAIHRcOZyakwOl1NZ4w8AAARCQUGB3G63EhMTTdsTExO1du3aao+59tprVVBQoPPPP1+GYaiyslK33nqr7rvvPv8+aWlpmjVrlrp3765du3Zp4sSJuuCCC/Tjjz+qWbNmVc5ZXl6u8vJy/+PCwkJJksvlksvlCsRb9XO5XP7Cn+ExAn7+cON7/009Dz7kw4x8mJEPM/JhRj7Mgp0P8hweKPwBANC4UPgLAjr+AABAqC1atEiPP/64XnrpJaWlpWnDhg2688479cgjj+jBBx+UJF122WX+/fv06aO0tDR17NhRb731lm6++eYq55w0aZImTpxYZfuCBQsUHR0dtPeyc+dOzZs3L2jnDyfZ2dmhDqFBIR9m5MOMfJiRDzPyYRasfJSUlATlvAgcwzhS+OvXL7SxAACAwKDwFwSs8QcAAAIpPj5eNptN+fn5pu35+flKSkqq9pgHH3xQ119/vf70pz9Jknr37q3i4mLdcsstuv/++2W1Vp3xvUWLFjr11FO1YcOGas85btw4ZWVl+R8XFhYqOTlZAwcOVFxcXG3fXrVcLpfef/N9SVKH9h2qTFHa1LhcLmVnZ2vAgAGy2+2hDifkyIcZ+TAjH2bkw4x8mAU7H77ZAcLR1KlT9fTTTysvL0+pqal64YUX1L9//2r3nTVrlkaPHm3a5nQ6VVZWVh+h1sn27dKePZLN5l3jDwAAhD8Kf8Hgq/vR8QcAAALA4XCoX79+ysnJ0bBhwyRJHo9HOTk5yszMrPaYkpKSKsU9m80m6djTkRcVFWnjxo26/vrrq33e6XTK6XRW2W6324NysdAXp81m4+LsYcHKdbgiH2bkw4x8mJEPM/JhFqx8hGuO586dq6ysLE2bNk1paWmaMmWKBg0apHXr1ikhIaHaY+Li4rRu3Tr/43BZo/jbb723p58uRUaGNhYAABAYVb/qjTrzdfyxxh8AAAiUrKwsTZ8+Xa+++qrWrFmjsWPHqri42P/t8lGjRmncuHH+/YcOHaqXX35Zc+bM0ebNm5Wdna0HH3xQQ4cO9RcA//rXv+qLL77Qli1b9PXXX2v48OGy2Wy65pprQvIej8Wi8LhwBgAAGofJkydrzJgxGj16tHr27Klp06YpOjpaM2fOPOYxFotFSUlJ/p9fr83cUH34off2ootCGwcAAAgcOv6CgDX+AABAoI0YMUJ79uzR+PHjlZeXp759+2r+/Pn+i0q5ubmmDr8HHnhAFotFDzzwgHbs2KE2bdpo6NCheuyxx/z7bN++Xddcc4327t2rNm3a6Pzzz9c333yjNm3a1Pv7q45xeN50q4XvqgEAgPpRUVGh5cuXm75QZbValZGRoSVLlhzzuKKiInXs2FEej0dnnnmmHn/8cZ1++un1EXKtVVQcKfz94Q+hjQUAAAQOhb8gYI0/AAAQDJmZmcec2nPRokWmxxEREZowYYImTJhwzPPNmTMnkOEFnEceSXT8AQCA+lNQUCC3212lYy8xMVFr166t9pju3btr5syZ6tOnjw4ePKh//OMfOvfcc/XTTz+pQ4cO1R5TXl6u8vJy/2Pfeogul0sulytA70b+cx596/PZZxYdPBihpCRDZ59dqQC/bIN1rHw0VeTDjHyYkQ8z8mFGPsyCnY+TOS+Fv2BgjT8AAIC6OzyUCpc1cgAAQNOUnp6u9PR0/+Nzzz1Xp512mv7v//5PjzzySLXHTJo0SRMnTqyyfcGCBYqOjg5KnNnZ2abHL77YV1JHnXHGFn322aqgvGZD9ut8NHXkw4x8mJEPM/JhRj7MgpWPkpKSGu9L4S8IrDbvdFQetyfEkQAAAIQv31SfdPwBAID6Eh8fL5vNpvz8fNP2/Px8JSUl1egcdrtdZ5xxhjZs2HDMfcaNG6esrCz/48LCQiUnJ2vgwIGKi4urXfDH4HK5lJ2drQEDBshut/u333uv97Lg//t/yRoypPrOxMboWPloqsiHGfkwIx9m5MOMfJgFOx++2QFqgsJfEFjthwt/Lgp/AAAAteUv/NHxBwAA6onD4VC/fv2Uk5OjYcOGSZI8Ho9ycnKOOeX6r7ndbq1evVpDhgw55j5Op1NOp7PKdrvdHrSLp0efu7hYWrfOu/2ccyLUFK/XBjPX4Yh8mJEPM/JhRj7MyIdZsPJxMuek8BcEvsKf2+UOcSQAAADhj44/AABQn7KysnTDDTforLPOUv/+/TVlyhQVFxdr9OjRkqRRo0apffv2mjRpkiTp4Ycf1jnnnKOuXbvqwIEDevrpp7V161b96U9/CuXbOK4ffpAMQ2rbVvrVcoYAACDMUfgLApvDJklyV1D4AwAAqC06/gAAQCiMGDFCe/bs0fjx45WXl6e+fftq/vz5SjxcIcvNzZXVavXvv3//fo0ZM0Z5eXlq2bKl+vXrp6+//lo9e/YM1Vs4oRUrvLdnnhnaOAAAQOBR+AsCm91b+GOqTwAAgNozDNb4AwAAoZGZmXnMqT0XLVpkevzss8/q2WefrYeoAofCHwAAjZf1xLvgZPkKf0z1CQAAUHdWC0NWAACAQKLwBwBA48VVlCDwr/HHVJ8AAAC15pF39gSm+gQAAAic8nLpp5+89yn8AQDQ+FD4CwJf4Y+pPgEAAOqOqT4BAAACZ9cuqbJScjql5ORQRwMAAAKNwl8Q2ByHp/qk4w8AAKDWDB1e44+OPwAAgIDZs8d726aNxDALAIDGh8JfEPjW+KPjDwAAoPYM43Dhj44/AACAgCko8N7Gx4c2DgAAEBwU/oLA6ji8xp+Ljj8AAIC6ouMPAAAgcI7u+AMAAI0Phb8g8HX8MdUnAABA7fmn+qTjDwAAIGDo+AMAoHGj8BcEVrs3rUz1CQAAUHus8QcAABB4dPwBANC4UfgLAjr+AAAA6s5X+LNaGLICAAAECh1/AAA0blxFCQJfxx9r/AEAANSeYTDVJwAAQKDR8QcAQONG4S8IbA5vxx9TfQIAANQeU30CAAAEHh1/AAA0bhT+goCpPgEAAAKHjj8AAIDAoeMPAIDGjcJfEDDVJwAAQN3R8QcAABB4dPwBANC4UfgLAt9Un3T8AQAA1B5r/AEAAARWZaW0b5/3Ph1/AAA0ThT+gsDX8ccafwAAAHVHxx8AAEBg+Ip+ktSqVejiAAAAwdMgCn9Tp05VSkqKIiMjlZaWpmXLltXouDlz5shisWjYsGHBDfAk+Tv+mOoTAACg1vxTfdLxBwAAEBC+9f1atZIiIkIbCwAACI6QF/7mzp2rrKwsTZgwQStWrFBqaqoGDRqk3bt3H/e4LVu26K9//asuuOCCeoq05mwRTPUJAABQV77Cn9US8iErAABAo8D6fgAANH4hv4oyefJkjRkzRqNHj1bPnj01bdo0RUdHa+bMmcc8xu12a+TIkZo4caI6d+5cj9HWDFN9AgAA1J2/44+pPgEAAALC1/HH+n4AADReIW3qr6io0PLlyzVu3Dj/NqvVqoyMDC1ZsuSYxz388MNKSEjQzTffrP/973/HfY3y8nKVl5f7HxcWFkqSXC6XXC5XHd+Bme98htV7kcpd4Q74a4QT33tvyjk4GvkwIx9m5MOMfJiRD7Ng54M8NxyGwVSfAAAAgUTHHwAAjV9IC38FBQVyu91KTEw0bU9MTNTatWurPWbx4sWaMWOGVq5cWaPXmDRpkiZOnFhl+4IFCxQdHX3SMdfEV998JUmqKKvQvHnzgvIa4SQ7OzvUITQo5MOMfJiRDzPyYUY+zIKVj5KSkqCcF7VHxx8AAEBgHP4+vJo3D20cAAAgeMJqGd9Dhw7p+uuv1/Tp0xVfw68mjRs3TllZWf7HhYWFSk5O1sCBAxUXFxfQ+Fwul7Kzs3XRJRdpjdbI4rFoyJAhAX2NcOLLx4ABA2S320MdTsiRDzPyYUY+zMiHGfkwC3Y+fLMDIPT8U33S8QcAABAQvu+4Bem78AAAoAEIaeEvPj5eNptN+fn5pu35+flKSkqqsv/GjRu1ZcsWDR061L/N4/GuoxcREaF169apS5cupmOcTqecTmeVc9nt9qBdPHVGe1/PU+lRREREk/+WejBzHY7Ihxn5MCMfZuTDjHyYBSsf5LjhYI0/AACAwCot9d5S+AMAoPGyhvLFHQ6H+vXrp5ycHP82j8ejnJwcpaenV9m/R48eWr16tVauXOn/+d3vfqff/OY3WrlypZKTk+sz/GOy2W3++x6XJ4SRAAAAhC86/gAAAALL1/EXFRXaOAAAQPCEfKrPrKws3XDDDTrrrLPUv39/TZkyRcXFxRo9erQkadSoUWrfvr0mTZqkyMhI9erVy3R8ixYtJKnK9lCyOY4U/twVbtNjAAAA1JC37kfHHwAAQIDQ8QcAQOMX8sLfiBEjtGfPHo0fP155eXnq27ev5s+fr8TERElSbm6urNaQNiaeNKv9SLxulzuEkQAAAIQvX8ef1RJeY0EAAICGio4/AAAav5AX/iQpMzNTmZmZ1T63aNGi4x47a9aswAdUR9aIIxenmOoTAACgdpjqEwAAILDo+AMAoPHj69NBYLFY/F1/7go6/gAAAGrDX/hjqk8AAICAoOMPAIDGj8JfkNjs3nX9mOoTAACgdgyDjj8AAIBAouMPAIDGj8JfkNgchwt/dPwBAADUCR1/AAAAgUHHHwAAjR+FvyDxTfXJGn8AAAC1wxp/AAAAgUXHHwAAjR+FvyBhqk8AAIC6YY0/AACAwKLjDwCAxo/CX5Aw1ScAAEBg0PEHAAAQGHT8AQDQ+FH4CxKm+gQAAKgbX8ef1cKQFQAAIBDo+AMAoPHjKkqQ0PEHAABQN4bBVJ8AAACBRMcfAACNH4W/IGGNPwAAgLrxr/HHVJ8AAAB15nJJlZXe+3T8AQDQeFH4CxKm+gQAAKgbf+GPjj8AAIA683X7SXT8AQDQmFH4CxKm+gQAAKgbOv4AAAACx7e+n8UiOZ2hjQUAAAQPhb8gYapPAACAOvLW/ej4AwAACABfx19UlLf4BwAAGicKf0FCxx8AAEDd0PEHAAAQOL6OP9b3AwCgcaPwFySs8QcAAFA3rPEHAAAQOGVl3jEV6/sBANC4UfgLEqb6BAAACAyrhSErAABAXdHxBwBA08BVlCBhqk8AAIC68RjemROY6hMAAKDufGv80fEHAEDjRuEvSJjqEwAAIDCY6hMAANS3qVOnKiUlRZGRkUpLS9OyZctqdNycOXNksVg0bNiw4AZYC77CHx1/AAA0bhT+gsTf8cdUnwAAALXiX+OPjj8AAFCP5s6dq6ysLE2YMEErVqxQamqqBg0apN27dx/3uC1btuivf/2rLrjggnqK9OT4pvqk4w8AgMaNwl+Q+Dr+mOoTAACgdvyFPzr+AABAPZo8ebLGjBmj0aNHq2fPnpo2bZqio6M1c+bMYx7jdrs1cuRITZw4UZ07d67HaGuurMx7S8cfAACNG4W/ILHZvR1/TPUJAABQN3T8AQCA+lJRUaHly5crIyPDv81qtSojI0NLliw55nEPP/ywEhISdPPNN9dHmLVSUuIdU9HxBwBA4xYR6gAaK/9Un3T8AQAA1AodfwAAoL4VFBTI7XYrMTHRtD0xMVFr166t9pjFixdrxowZWrlyZY1fp7y8XOXl5f7HhYWFkiSXyyWXy3XygR+H73xFRW5JNjmdHrma8NI0vnwEOs/hinyYkQ8z8mFGPszIh1mw83Ey56XwFyT+qT6b8EAKAACgLgyDNf4AAEDDdujQIV1//fWaPn264uPja3zcpEmTNHHixCrbFyxYoOggteT9+OMmST20Z89WzZu3KiivEU6ys7NDHUKDQj7MyIcZ+TAjH2bkwyxY+SjxLdZbAxT+goSpPgEAAALDamF2egAAUD/i4+Nls9mUn59v2p6fn6+kpKQq+2/cuFFbtmzR0KFD/ds8Hu+1oIiICK1bt05dunSpcty4ceOUlZXlf1xYWKjk5GQNHDhQcXFxgXo7krwdAtnZ2WrXzhtHjx6naMiQDgF9jXDiy8eAAQNkt9tDHU7IkQ8z8mFGPszIhxn5MAt2PnyzA9QEhb8gYapPAACAuvHIe9GMqT4BAEB9cTgc6tevn3JycjRs2DBJ3kJeTk6OMjMzq+zfo0cPrV692rTtgQce0KFDh/Tcc88pOTm52tdxOp1yOp1Vttvt9qBdPC0v936ZKjbWJvvhL6w3ZcHMdTgiH2bkw4x8mJEPM/JhFqx8nMw5KfwFCVN9AgAA1I1/jT+m+gQAAPUoKytLN9xwg8466yz1799fU6ZMUXFxsUaPHi1JGjVqlNq3b69JkyYpMjJSvXr1Mh3fokULSaqyPdRKS71jqqioEAcCAACCisJfkPg6/jwVTPUJAABQK966Hx1/AACgXo0YMUJ79uzR+PHjlZeXp759+2r+/PlKTEyUJOXm5spqDb+pyH1LAwVpCUEAANBAUPgLEt8af3T8AQAA1A4dfwAAIFQyMzOrndpTkhYtWnTcY2fNmhX4gAKgtNR7S8cfAACNW/h9PSlM+Kb69Ljo+AMAAKgNf+GPjj8AAIA68xX+6PgDAKBxo/AXJL6pPt0VdPwBAADUBR1/AAAAdUfHHwAATQOFvyBhqk8AAIC6oeMPAAAgcHxr/FH4AwCgcaPwFyR0/AEAANSNYXgLf1YLQ1YAAIC6crm8X6ZyOkMcCAAACCquogQJa/wBAADUjb/jj6k+AQAA6szl8t5GRIQ2DgAAEFwU/oKEqT4BAADqhqk+AQAAAqey0ntrt4c2DgAAEFwU/oKEqT4BAECgTZ06VSkpKYqMjFRaWpqWLVt23P2nTJmi7t27KyoqSsnJyfrLX/6isrKyOp0zFOj4AwAAqDtf4Y+OPwAAGjcKf0Fij/F+faqiqCLEkQAAgMZg7ty5ysrK0oQJE7RixQqlpqZq0KBB2r17d7X7z549W/fee68mTJigNWvWaMaMGZo7d67uu+++Wp+zvtHxBwAAEDi+qT7p+AMAoHGj8Bckkc0jJUnlB8tDHAkAAGgMJk+erDFjxmj06NHq2bOnpk2bpujoaM2cObPa/b/++mudd955uvbaa5WSkqKBAwfqmmuuMXX0new565thsMYfAABAoNDxBwBA08Cv+iBxNndKksoOlp1gTwAAgOOrqKjQ8uXLNW7cOP82q9WqjIwMLVmypNpjzj33XL3xxhtatmyZ+vfvr02bNmnevHm6/vrra33O8vJylZcf+VJTYWGhJMnlcsnl+wp5gBx9PrfbHfDzhxvf+2/qefAhH2bkw4x8mJEPM/JhFux8kOeGhY4/AACaBgp/QeLr+HOXu1VZXqkIJ6kGAAC1U1BQILfbrcTERNP2xMRErV27ttpjrr32WhUUFOj888+XYRiqrKzUrbfe6p/qszbnnDRpkiZOnFhl+4IFCxQdHV2bt3Zcvqk+V6xYIftGrlBJUnZ2dqhDaFDIhxn5MCMfZuTDjHyYBSsfJSUlQTkvaoeOPwAAmgZ+1QeJo5nDf7/8YLkiEkg1AACoP4sWLdLjjz+ul156SWlpadqwYYPuvPNOPfLII3rwwQdrdc5x48YpKyvL/7iwsFDJyckaOHCg4uLiAhW6JG+HgLHeW/g7+6yzNaTbkICeP9y4XC5lZ2drwIABsvM1ffLxK+TDjHyYkQ8z8mEW7Hz4ZgdAw0DHHwAATQPVqCCx2qxyNHOo4lCFyg6WKSYhJtQhAQCAMBUfHy+bzab8/HzT9vz8fCUlJVV7zIMPPqjrr79ef/rTnyRJvXv3VnFxsW655Rbdf//9tTqn0+mU0+msst1utwflYqFvjT97RHDOH46CletwRT7MyIcZ+TAjH2bkwyxY+SDHDYuv448/FgAAGjdrqANozHzTfZYfLD/BngAAAMfmcDjUr18/5eTk+Ld5PB7l5OQoPT292mNKSkpktZqHejabTZK3oFabc9Y331SfFllCHAkAAEB4MwzJ5fKOqZjqEwCAxo1f9UEU2SJShdsLVXagLNShAACAMJeVlaUbbrhBZ511lvr3768pU6aouLhYo0ePliSNGjVK7du316RJkyRJQ4cO1eTJk3XGGWf4p/p88MEHNXToUH8B8ETnDDV/4c9C4Q8AAKAuPJ4j9+n4AwCgcaPwF0TO5t6psMoOUvgDAAB1M2LECO3Zs0fjx49XXl6e+vbtq/nz5ysxMVGSlJuba+rwe+CBB2SxWPTAAw9ox44datOmjYYOHarHHnusxudsKOj4AwAAqBu3+8g4kY4/AAAaN37VBxFTfQIAgEDKzMxUZmZmtc8tWrTI9DgiIkITJkzQhAkTan3OUKPjDwAAIDCOLvzR8QcAQOPGGn9BRMcfAABA7RkGa/wBAAAEgtt9ZDxFxx8AAI0bhb8g8hX+6PgDAACoPTr+AAAA6qayksIfAABNRYMo/E2dOlUpKSmKjIxUWlqali1bdsx9p0+frgsuuEAtW7ZUy5YtlZGRcdz9Q8k31ScdfwAAACfPN9Wn1dIghqwAAABhy+PxjqdsNonvVAEA0LiF/CrK3LlzlZWVpQkTJmjFihVKTU3VoEGDtHv37mr3X7Roka655hr997//1ZIlS5ScnKyBAwdqx44d9Rz5idHxBwAAUHv+Nf6Y6hMAAKBOfB1/rO8HAEDjF/LC3+TJkzVmzBiNHj1aPXv21LRp0xQdHa2ZM2dWu/+bb76p2267TX379lWPHj30r3/9Sx6PRzk5OfUc+YlFtvB2/FH4AwAAOHn+wh9fSwcAAKgT3xp/TPMJAEDjF9LCX0VFhZYvX66MjAz/NqvVqoyMDC1ZsqRG5ygpKZHL5VKrVq2CFWat+af6PMBUnwAAACfLMOj4AwAACAS323sJkI4/AAAav5B+z6egoEBut1uJiYmm7YmJiVq7dm2NzvH3v/9d7dq1MxUPj1ZeXq7y8iMdd4WFhZIkl8sll8tVy8ir5zuf7zYixpve0gOlAX+tcPDrfDR15MOMfJiRDzPyYUY+zIKdD/Lc8NDxBwAAUDe+jj8KfwAANH5h3eD/xBNPaM6cOVq0aJEiIyOr3WfSpEmaOHFile0LFixQdHR0UOLKzs6WJBX9XCRJ2rdrn+bNmxeU1woHvnzAi3yYkQ8z8mFGPszIh1mw8lFSUhKU8+LkscYfAABAYPg6/pjqEwCAxi+kv+7j4+Nls9mUn59v2p6fn6+kpKTjHvuPf/xDTzzxhD7//HP16dPnmPuNGzdOWVlZ/seFhYVKTk7WwIEDFRcXV7c38Csul0vZ2dkaMGCA7Ha7difv1ob7NiiiMkJDhgwJ6GuFg1/no6kjH2bkw4x8mJEPM/JhFux8+GYHQOixxh8AAEBg0PEHAEDTEdLCn8PhUL9+/ZSTk6Nhw4ZJkjwej3JycpSZmXnM45566ik99thj+uyzz3TWWWcd9zWcTqecTmeV7Xa7PWgXT33njo2PlSSVHyxv0hdqg5nrcEQ+zMiHGfkwIx9m5MMsWPkgxw0PHX8AAAB14yv80fEHAEDjF/Jf91lZWbrhhht01llnqX///poyZYqKi4s1evRoSdKoUaPUvn17TZo0SZL05JNPavz48Zo9e7ZSUlKUl5cnSYqNjVVsbGzI3kd1nM29BUd3hVuVZZWKiAx5ugEAAMKGYXg7/qwWa4gjAQAACG++qT75jhsAAI1fyCtRI0aM0J49ezR+/Hjl5eWpb9++mj9/vhITEyVJubm5slqPXOx5+eWXVVFRod///vem80yYMEEPPfRQfYZ+Qs5mTskiyZDKDpYpNrJhFSYBAAAaMo88kpjqEwAAoK4qK+n4AwCgqWgQv+4zMzOPObXnokWLTI+3bNkS/IACxGK1yBnnVPnBcpXtL1NsIoU/AACAk8VUnwAAAHVDxx8AAE0H8yYFWbO2zSRJhTsKQxwJAABAeDHkneqTjj8AAIC6YY0/AACaDgp/QdaiUwtJ0oHNB0IaBwAAQLjxrfFHxx8AAEDd0PEHAEDTQeEvyFp2bilJ2r9pf4gjAQAACE90/AEAANQNHX8AADQdFP6CjI4/AACA2vFP9UnHHwAAQJ1UVnrHU3T8AQDQ+FH4CzI6/gAAAGqHNf4AAAACw+Nhqk8AAJoKCn9B1rIThT8AAIC6sFoYsgIAANSFr+OPqT4BAGj8uIoSZL6pPksKSlR+qDy0wQAAAIQRj+GRxFSfAAAAdeXxMNUnAABNBYW/IItsHqmoVlGSWOcPAACgNpjqEwAAoG4qK72XAOn4AwCg8aPwVw9Y5w8AAODk+df4o+MPAACgTtxuOv4AAGgqKPzVA3/hbzOFPwAAgJryF/7o+AMAAKgTt5uOPwAAmgoKf/Wg1amtJEn5P+SHOBIAAIDwQ8cfAABA3VRW0vEHAEBTQeGvHnS8oKMkacuiLaENBAAAIIwYBh1/AAAAgeDxeMdTdPwBAND4UfirB8nnJssaYdXBrQd1YMuBUIcDAAAQFljjDwAAhMrUqVOVkpKiyMhIpaWladmyZcfc97333tNZZ52lFi1aKCYmRn379tXrr79ej9GeWGWl9xIgHX8AADR+FP7qgSPWoXZnt5Mkbf7v5hBHAwAAEF6sFoasAACg/sydO1dZWVmaMGGCVqxYodTUVA0aNEi7d++udv9WrVrp/vvv15IlS7Rq1SqNHj1ao0eP1meffVbPkR8bHX8AADQdXEWpJym/SZEkbV20NbSBAAAAhAmPPJKY6hMAANSvyZMna8yYMRo9erR69uypadOmKTo6WjNnzqx2/4svvljDhw/Xaaedpi5duujOO+9Unz59tHjx4nqO/Njo+AMAoOmg8FdPUi5OkSRtzN4oj9sT2mAAAADCgH+NP6b6BAAA9aSiokLLly9XRkaGf5vValVGRoaWLFlywuMNw1BOTo7WrVunCy+8MJihnhS32zueovAHAEDjR4N/Pel4YUdFtoxU0a4ibV64WV0GdAl1SAAAAGGBjj8AAFBfCgoK5Ha7lZiYaNqemJiotWvXHvO4gwcPqn379iovL5fNZtNLL72kAQMGHHP/8vJylZeX+x8XFhZKklwul1wuVx3fhZnL5ZLb7f3uv8XilsvVtL+Q7stvoPMcrsiHGfkwIx9m5MOMfJgFOx8nc14Kf/UkwhmhXlf30ncvf6cfXv2Bwh8AAMAJGKLjDwAAhIdmzZpp5cqVKioqUk5OjrKystS5c2ddfPHF1e4/adIkTZw4scr2BQsWKDo6OuDxud2pkqTNm9dr3rz1AT9/OMrOzg51CA0K+TAjH2bkw4x8mJEPs2Dlo6SkpMb7UvirR6k3pOq7l7/TmvfWqLywXM44Z6hDAgAAaLD8hT86/gAAQD2Jj4+XzWZTfn6+aXt+fr6SkpKOeZzValXXrl0lSX379tWaNWs0adKkYxb+xo0bp6ysLP/jwsJCJScna+DAgYqLi6v7GzmKy+XSCy/skST17HmqhgzpGtDzhxuXy6Xs7GwNGDBAduY+JR+/Qj7MyIcZ+TAjH2bBzodvdoCaoPBXj9r3b6/40+JVsKZA30z5RheNvyjUIQEAADR4dPwBAID64nA41K9fP+Xk5GjYsGGSJI/Ho5ycHGVmZtb4PB6PxzSV5685nU45nVW/EG6324NysdA31WdkpE12uy3g5w9Hwcp1uCIfZuTDjHyYkQ8z8mEWrHyczDmtAX91HJPFYtFFE7zFvq+e+kpFeUUhjggAAKDhouMPAACEQlZWlqZPn65XX31Va9as0dixY1VcXKzRo0dLkkaNGqVx48b59580aZKys7O1adMmrVmzRs8884xef/11XXfddaF6C1W43d7xVAQtAAAANHr8uq9np//xdH3z7DfasXSHvnj4C13+0uWhDgkAAKBBMgxv4c9q4btqAACg/owYMUJ79uzR+PHjlZeXp759+2r+/PlKTEyUJOXm5spqPTI+KS4u1m233abt27crKipKPXr00BtvvKERI0aE6i1UUVnpjZeGDAAAGj8Kf/XMYrFowFMDNOuiWVrxrxU6/97z1fyU5qEOCwAAoMHxd/wx1ScAAKhnmZmZx5zac9GiRabHjz76qB599NF6iKr26PgDAKDp4OvTIdDxwo7qdEkneVwe/e/x/4U6HAAAgAaJqT4BAAACw1f4o+MPAIDGj8JfiFz0kHetv+9nfK89a/aEOBoAAICGi44/AACAunG7vZcA6fgDAKDxo/AXIh0v6Kjuv+suT6VH8++Y71/DBgAAAF6+8REdfwAAAHVDxx8AAE0Hhb8QGvTsINmcNm36fJM+veNTuSvcoQ7JpLK8Uu+Pel8rX10Z6lAAAEATxBp/AAAAgUHhDwCApoPCXwi17NxSA54aIEn69sVv9cbgN1RRVOF/vmRvifZv3h+q8LQpe5NWvb5K/33wvyGLAQAAgI4/AACAumGqTwAAmg4KfyGWdkearvn4GjmaObTlv1v0xuA3VF5YrnUfrdMLXV/QSz1f0oGtB0ISW/6qfElS4bZCuUpdIYkBAAA0XXT8AQAABAYdfwAANB0U/hqAU397qq7Pvl6RLSK17atter7r85pzxRyVHShTZVmlVr+5OiRx7V6923//wOYDIYkBAAA0Xb7Cn9XCkBUAAKAuKivp+AMAoKngKkoD0SGtg0YtHKWo1lEq2VMii82iyBaRkqRVr6+SYXgvfG34bIOeaPGEVvxrRdBjyl+d77+/b+O+oL8eAACAj2/sIzHVJwAAQF15PHT8AQDQVFD4a0DantFWNy+5WZdOulR/Xv9n3bX1LkVERqhgbYF2Ld8lwzCUc2+Oyg+WK/uebJXuLw1aLJXlldq7bq//8b4NFP4AAED98XX7SUz1CQAAUFd0/AEA0HRQ+GtgWndrrfPvPV8tO7eUM86p7ld0lyT9e+i/tfD+hcpbmSdJKjtQpq+e/CpocRSsLZCn0uN/vH/j/qC9FgAAwK/R8QcAABA4rPEHAEDTQeGvgct4IkNterZRUV6RFk9aLElK6pskSfrqqa/01dNfmS6MBcrR6/tJdPwBAID6RccfAABA4PgKf3T8AQDQ+FH4a+BapLTQmO/G6OKJF6tVt1aKS47T1R9erbPGniUZ0ud/+1wf3fSRKssqA/q6+au86/u1Ob2NJAp/AACgftHxBwAAEDhut/cSIB1/AAA0fnzPJwzYo+y6aPxFumj8Rf5tQ6YOUfxp8frsrs+0ctZKrXlvjRL7JMpisyipb5I6nNNBHS/sqOj4aK2ctVIVRRXqeFFHtevXrkavuXnhZklS72t7a+H9C3Vw60G5XW7Z7LagvMe68k1NmtArIdShAACAAKDjDwCaFrfbLZfLFfTXcblcioiIUFlZmdxud9Bfr6Graz5sNpsiIiL4kk4YoOMPAICmg1/3YcpisSjtz2mK7x6vj/70kQq3FSp3ca4kaesXW7X0uaWyRljVvGNz0/p8w14bptTrU4977qL8Iu1avkuS1Hd0X335yJeqLKtU3so8tT+7ffDeVC2VHSjTv9L+JY/bo7u23qXo1tGhDgkAANQRHX8A0HQUFRVp+/btQVnG4tcMw1BSUpK2bdvG7xcFJh/R0dFq27atHA5HgKNDIFVW0vEHAEBTQeEvzHUZ2EV3bblLO5btUOH2QrlKXNrx7Q5t+2qb8r7P0/6N+xXZMlJtz2yrzTmb9fGfPpbhNtTnuj6yRlQ/0+vGzzZKktqe2VbN2jZTym9StOHTDZpzxRyNeH+EOqR1qM+3eEKr/71a5YXlkqTNOZt1+h9PD3FEAACgro7u+LNamJ0eABort9ut7du3Kzo6Wm3atAl6Mc7j8aioqEixsbGyWvn9Upd8GIahiooK7dmzR5s3b1a3bt3IaQPm8Xj/blH4AwCg8aPw1whYrBZ1OOdIMS51lLejb9uSbdowf4POuOkMNU9urrf/+LbWvLtGH47+UPPvmq82PdsoqlWUWndvLXuUXbtW7NK2r7ep/KC3iNZlcBdJ0vDXhuvVS17V7tW7NeOcGep1dS9d9uJlDaazbuXMlf77GxdspPAHAEAj4DE8/vtM9QkAjZfL5ZJhGGrTpo2ioqKC/noej0cVFRWKjIykSKW65yMqKkp2u11bt271nwcNU2UlU30CANBU8Ou+EUtOT1ZyerL/8ZVvXqkl/ZZoyTNLVLq3VNuXbJck/fLJL9Ue3+2ybpKk6Pho3bDwBi346wKten2VfpzzozblbFLvkb2VfG6y2vRso9bdWsvmqP/1//JX5Wvndzv9jzd+tlGGYTBly2Ful1tfPvKlEvskqufve4Y6HAAAaoypPgGgaeHf+vBFATU8eDxM9QkAQFNB4a8JiXBG6IJxF+jcv56rPT/v0b5f9ql0X6nyV+fLU+lR61Nbq/3Z7bX636sV4YxQ8rlHiobR8dEaNmuY+v+5v96/7n0VrC3Q0ilLtXTKUkmSxWZR626t1aZnG8W2jVV5Ybn2/LxHpdGlWpm3UpXFlUo+L1kJvRIkQzI8hgyPIUes45hTjtbE0ue9r99tSDdtytmkwu2FKlhboDantTHtt2/DPu1dv1ddL+ta6/9Qlh8q1ysXvKKYNjH6w9t/UGSLhv9NxsVPLNaXj3ypiKgIdRnURc5mzlCHBABAjRw91ScdfwAAALXn8RyZ6pOOPwAAGj9+3TdBNrtNSalJSkpNqvb5U84/5ZjHtuvXTv9v5f/ThvkbtP7j9dr9427t+XmPKg5VqGBtgQrWFlQ5Zt7/5h3zfBFREWrTs42ccU7Zo+2yR9vliHWoWbtmkkWyR9mV1DdJ0fHRcjRzyBHrkDPOKWecU0V5RVr1+ipJ0gX3XyC3y61N2Zv06Z8/1dUfXi1HjHdh8UM7D2nGuTNUsqdEv/2/36rfLf1qnKuyA2X+4uT3M79X/g/5kqTXMl7T9QuuV1SrqlPR7NuwT9/P/F7n3HWOYhJiavQ6B3MPKrZtrGz2wHVN5q/O15ePfClJqiyt1NoP1ir1+tSAnR8AgGCi4w8AACAwKiuP3KfjDwCAxo/CH05ahDNCPa7ooR5X9JDkvTB3aMch7VmzR3t+2qPSfaWKiIxQXEqcFr+xWM0qm8kR5dDmhZtVUVRhOldlaaV2Ld910jE4mjkUERkhd4VbyecmK/ncZF088WJt+3qbNuds1jNtn1HKRSlqn9Ze6z5ap5I9JZKk+XfOl8VqUVxynGwOmzqkdZA9+sio113h1pr31qhgbYEO5h7UqtdXqXnH5vrt//1Wy55fJsnb3bhr+S69+ptXdd2C6xSbGOs/vuxAmd4Y/Ib2b9yv7Uu26/rPr5fVdvyOxvWfrNe/h/5bnS7ppOvmX1enDsijLbxvoTwujxyxDlUUVWj1m6sp/AEAwgYdfwCAcLBkyRKdf/75Gjx4sD755JNQhwNUy+U6cp+OPwAAGj9+3aPOLBaL4jrEKa5DnLoM6OLf7nK5tCVmi4YMGSK73S5PpUeVZZWyWC2yWC2SRTqw5YD2rtsrV4lLrlKXXCUulR8sV+GOQlksFpXuK9Wen/ao7GCZKg5VqKKoQu4Kt/f+IW8R8YIHLpDkXdPw+uzr9fbv39ahnYe0/j/rtf4/6yVJjliHEvskatvX2/TxmI/9MdocNsUkxMgeY5c1wqrCbYUqLyw3vb/9G/fr9YzXJUmRLSN13WfX6d9D/638VfmaetpUnXPXOWrbr60qyyr1zbPfaP/G/ZKkLYu26O3fv61TLjxFPYb1UEy7qt1/7gq3PrvrM8mQNuds1ie3f6JLHr1EMW2O3SnoqfRoY/ZGuYpdap/WXs2Tm1fZJ39Vvtb/Z70sVot+/9bvNXvIbG3K3qTt32xXh3M6yFXq0qGdh9SyU0vvn0WQuUpcpgIrAAAnQscfACAczJgxQ3/+8581Y8YM7dy5U+3atQtJHBUVFXI4HCF5bTR8Rxf+6PgDAKDxo/CHemONsMoRa/6PSHz3eMV3jz+p87hKXTq49aDKDpTJ2dxpWs8vOT1Zd+XepfxV+dq8cLPyV+YrJilGfa7roxYdW+irp75S7uJcVRyqUElBiQq3F6pwe6Hp/M3aNVPXIV1ltVl12pWn6ed3ftbqN1fLVeLSOX85R+3Pbq/RX47W2394W/mr8rVowiLT8RGREep3az8tnbJUaz9Yq7UfrNWCrAXeHERZtbHVRkU2j5QzzilXiUv7NuyTo5lDFYcqtOKfK7Ri+gol9EpQy84tZbPbVH6oXBWHKlR+qFyVZZUqO1Dm72CUpC6DuqjtmW0VERkhi9WiQ7sOKffLXElSz9/3VLfLuqnTJZ20eeFmzTxvpk654BTt/nG3SveWKjYpVqdffbr6XNdHCacnyGKzyBphrdEFVneFWytnrdRPc3/S3l/2qvOAzuozso86XtRRVptVhmFozXtrtHTKUuUuzlWvq3vp/PvOV2TzSDU/pWqxsiExPIb2/rJXrbu1rpfCKACgKjr+AAANXVFRkebOnavvvvtOeXl5mjVrlu677z7/8x9//LEefvhhrV69WrGxsbrgggv0/vvvS5LKy8s1fvx4zZ49W7t371ZycrLGjRunm2++WbNmzdJdd92lAwcO+M/1wQcfaPjw4f4vxjz00EP64IMPlJmZqccee0xbt26Vx+PR/Pnz9eijj+rHH3+UzWZTenq6nnvuOXXpcuRLutu3b9c999yjzz77TOXl5Tr11FP10ksvqW3bturcubOWLVums846y7//lClT9Oyzz2rz5s2yWgMzQw3q19FTfdLxBwBA48eve4Qde5Rd8T2OXSy02qxqe0ZbtT2jbZXnLn38Uv99wzB0YMsBle4r9XcSxibGKv60eNNae10GdtGQF4fo4LaDatmppSSp9amtdcvyW7Ty1ZXa+NlG7ftln2SRks9LVv/M/mp9amt1HdxV277eptz/5WrLoi2SIXlKPTq045AO7Thkimvwc4NljbBq6ZSl2rVil3av3q3dq3cf8z1Gx0er+SnNtev7Xdr42UZt/GxjlX0sVovOH3e+JOkPb/9Bn97xqVa/uVpbv9h6eAepKK9IS6cs1dIpS/3HRURFKLJFpL8z02K1yGKxmB9bvd2YxbuL/cetnLlSK2euVGSLSLXq2koVRRWmNR9/nPOjfpzzoySpff/2atW9lbZv267333xfjmiHyvaXqXBHocoPlqvN6W3kKnGpdG+pIltEKrJFpJwtnP77kS0iZY2wqmhXkXYs26GYNjGKbRsrV6lLzmZOOZs7Fdk8UvYYu8oOlHmnnm0fp2btmikiMkKHdh5SUX6RWndrLUkq2VsiV4lL0a2jVXawTIsnLdb2JdvV+tTWSr87XX2u7yN7lPlrkR63p8o0rnk/5Gn36t1yxDqU8psURTaPPOaf4Yn4/kPvK8IWrCtQ7uJcJfZOVFxynGLaxPinhfW4Pf4/o6bMU+nR8n8uV0LvBHW8oGOow2lQDMNQ4bZCxXWIa/KfE4SPozv+rBYuMgJAU2EYhkpcJSfesZY8Ho+KXcWyVdiqFLGi7dEn1WX+1ltvqUePHurevbuuu+463XXXXRo3bpwsFos++eQTDR8+XPfff79ee+01VVRUaN68ef5jR40apSVLluj5559XamqqNm/erIKCguO8WlUbNmzQu+++q/fee082m/f/sMXFxcrKylKfPn1UVFSk8ePHa/jw4Vq5cqWsVquKiop00UUXqX379vroo4+UkJCgr776Sh6PRykpKcrIyNArr7xiKvy98soruvHGGyn6hTFfx5/VasjK/wcAAGj0GkThb+rUqXr66aeVl5en1NRUvfDCC+rfv/8x93/77bf14IMPasuWLerWrZuefPJJDRkypB4jRmNgsVjUslNLfzHveGwOm1p1aWXaZo2w6sybz9SZN59Z7TFdB3VV10FdJUmVZZU6VHBI2R9n65wzz5G7xK3yg+UyDEMxCTFKPjdZFotFqdenqiivSNu/2a6i/CLvGn3NHHI2c8oR61BEVIS3sHm4w2//5v1a/eZqlewtUWVppTxuj2ITY9WsfTO1PbOtkvomSZKiWkXpyjeu1IUPXKjcxbmKSYhR54zO2rxws1ZMX6GtX25V2YEyb6yllSoqLapRDmPbxio9K11tTm+jte+v1c9v/6yyA2Xa+d1OSd4iYnpWuk45/xQtfGChDmw+oLKDZdqxbId2LNshSdqv/VXOu2/Dvhq9frDtXb9X//l//1H2PdlK7JOo2KRYlR0s0971e3Uw96CiW0crqnWUPJUe2aPs2v3jkWJtRGSE2vRsI1mk8sJyRbaIVGxirKLjoyWrlPd9nsoLy9WsbTPFto1VdGK08vbm6YtlX2j3yt3asWyHXKUudRvSTc3aNdN3076Tu9ztP39MQox6DO+hXct3KX91vhyxDvX/c391HdRVzU9pLpvDpkO7Dmnjgo2yRliV2CfRW9C2eP+Myw6UqexgmawRVtmj7Ud+ory3EZERMgxDnkqPPJUeGe4j92WRolpGKap1lKJaRkmS9m3cJ8NjeAuvcd7P69EFJrfLrYqiCrmKXSrdX6pdK3ap4lCF4pLjlHLxiYukleWVOrD5gA7mHpQzzqlW3VopunW0/3lPpUfvXfeefpr7kyw2i66YeYV6DO8hZzNnoD4OYWvD/A3KuS9Hed/nqfOAzhrx/gg5YgI/DVRlWaXKD5Ufd6piH8MwtOyFZVo9e7UynshQysUpAY8H4c9jePz3meoTAJqOEleJYifFnnjHICgaV6QYx4nHMj4zZszQddddJ0kaPHiwDh48qC+++EIXX3yxHnvsMV199dWaOHGif//UVO+a6+vXr9dbb72l7OxsZWRkSJI6d+580vFWVFTotddeU5s2R2bBueqqq0z7zJw5U23atNHPP/+sXr16afbs2dqzZ4++/fZbtWrVSh6PRwkJCYqLi5Mk/elPf9Ktt96qyZMny+l0asWKFVq9erU+/PDDk44PDYev449uPwAAmoaQ/8qfO3eusrKyNG3aNKWlpWnKlCkaNGiQ1q1bp4SEhCr7f/3117rmmms0adIk/fa3v9Xs2bM1bNgwrVixQr169QrBOwBOLCIyQrGJsXK2c6rtmW1lP86k+rFJseoxrEeNztuyU0td+MCFNY4jvke8qVvy1N+eqlN/e6oMw/AX/sr2ewtCMrxTXhoeQ4ZhHLl/+Mditahdv3b+tfu6XdZNQ14cooK1BTqw5YBkkdqf3V6xSd7/tHcd7C2CFuUXaf3H61W0u0jr1q9Tzz49ZbgMRbaIVLN2zWSP9hbQHLEOxSbFqryw3FukOvpnf5kMtyFnnFPt+rdT6d5Sle4rVURUhCqKKlR+sFzlB8tVUVyhyBaRcpW4vJ2WOw/J7XIrqlWUYpNite+XfbI5bIpqFSV7tF0lBSVyxDqU1DdJ5/7tXG1csFHfPPuNDm49qNzFuVXyWVJQopKCI99GttgsOuX8U1S0q0h71+/VrhW7Tvhn4lsT0idPeabHP7/9s/9+Qu8EFe0q8ndbLv+/5f7nSstL9cVDX+iLh7444WsGmjXC6i0I/ooj1iGr3SpXsUvuCnc1Rx4+3m5Vi5QWimoV5f9xlbq0ff12Tb9/ukr3laoor0hHzfonSWrdvbXa9GwjZzOntv5vqw5sPiBJMtyGPrjhA+kGb9G7RacWatmppVp0aqEWKS3kbO5U+cFyHdhyQJVllUe6ig7fVJZXev8eHPAWRqNbRyuyZaQMw5C7wi1PhUfuCrfcFW5ZI6yKio/ydmDard6/H25DHrdHEZHeQn3x7mIVrC3wTncc41Du4lzZY+xHYjocX3R8tNwut9zlbv/5fT8VpRXavXK3vln7jWLbxHqLxfHRMtyG3C63PC6P6bayrFKuYu9Uwl8//bUMj/fNbcrepOlnT1ef6/ro1KGnKiYhRja7TZEtI6sUVgzDUGVZpSoOVaiyrFKVZZXeIrBheP99OCpvhdsL9cltn+jg1oNKuThFp1x4ijpe0FEpF6f4O1N9CrcXav5d87Xm3TWSpNm/na2rZl+lbpd3q9JFK3mnd/Z9fsoLyxXhjJA1mm+bNwVM9QkAaMjWrVunZcuW+afujIiI0IgRIzRjxgxdfPHFWrlypcaMGVPtsStXrpTNZtNFF11Upxg6duxoKvpJ0i+//KLx48dr6dKlKigokMfjHafn5uaqV69eWrlypc444wy1atWqulNq2LBhuv322/X+++/r6quv1qxZs/Sb3/xGKSkpdYoVoeXr+GN9PwAAmoaQF/4mT56sMWPGaPTo0ZKkadOm6ZNPPtHMmTN17733Vtn/ueee0+DBg3XPPfdIkh555BFlZ2frxRdf1LRp0+o1dqCxsFgs/s4t321t2Bw2JfZJVGKfxGPuE5sYqzP/dKZcLpf2z9uv/kP6VymEdr705L/tGgxtTmuj/rf3156f92j3T7u9hcEYh1qf2lotUlqoeHexyg+Vy2qzqmRvidqe0VbNT2kuwzC0e/VuFW4vlGF4C5Rl+8tUvLtYJXtL5K5wq81pbRSTGKOiXUU6tOuQCrcXav2q9UrukKyk1CS1T2svSdq4YKMO7Tik9mnt1ffGvrJYLHK73PrprZ+0Y+kOtTu7nZLTk7Xzu5364bUflPd9nor3FMtwG7JH29Xxwo6y2Cza98s+f6HG5rApqmWUnHFOGR5DrhJX1Z9Sl6w2q3fNx8NrP1ojrN71Gz2GSveXqvxguSRvt509xi6bw6aKQxX+ImBFUUWVnFojrHI0cyjh9ATFJMRo90+7tXfdXu90udUo0pHuU3uMXS1SWqjiUIUO5h7U3nV7tXfdXv/zkS0i9buZv1Pu4lz9MOsHle4r9f/sWn7iImx9Ozr2mtqpnbV6rTNuPkM9f99T717zrgrWFGjh/Qu18P6F/uft0XZZ7YfX97RIMrx/ftUVdE9ky6It3umND583JiFGUa2iZHPaVHagTPt+2SdPpUfWCKva9Gyj/FX5mnPFHEVERvgLv5EtI+Wp9KhgTYH/Swm/ZrFbFP94vM7763m1SQnCwNFTfdLxBwBNR7Q9WkXjajYDSW14PB4VHipUXLO4aqf6rKkZM2aosrJS7dq1828zDENOp1MvvviioqKO/f+q4z0nSVar1fR7UJJcvsrNUWJiqnYnDh06VB07dtT06dPVrl07eTwe9erVSxUVFTV6bYfDoVGjRumVV17RlVdeqdmzZ+u555477jFo+HwdfxT+AABoGkJa+KuoqNDy5cs1btw4/zar1aqMjAwtWbKk2mOWLFmirKws07ZBgwbpgw8+CGaoAJoo3zSZ1RUz4zrEVXuMxWI5YQH011wul0rnlWrIkCGmQmiHtA5V9rXZbeozso/6jOzj39aqayv1utrb9ezr0KyueyqQPJUele4vVWVZpeLae9eOMwxD7nK3ygvLVX6oXO4KtxyxDu9PjEM2h63KefZv3q/C7YWmQp0lwqJ129bp3EvPVbPEZmrWvpliEmL8F/9LCkq0c/lO7ftln1ylLjVr10ynDT9N9mi7Tht+mgY9M0jlhd6uvv2b9+vA5iO3rhKXHLEOtUhpIUfs4SkvLfIXvXwdcJEtIuVxebwx7S+VxWqRzWGTzWFThDNCNodNleWV/s5PT6VHFqtFVptVFqtFleWVMioNRbWOUqturfznOuWCUyRD/nh8sZUdKPOf/9c/lgiL8gvy1SG5g8r2l6loV5FK9pbIGmGVzW6T1W6+tTltcsQ4ZI+xq/OAzjrjpjNksViUuT5Ta99fq3UfrdOmzzf5p491lVS9kHQ0m9Pm7WCMOFIc9BdiLN41Rbtd3k3n3HmONi/crF0rdmn9f9ardG+pDmw54O0CPkryecm67PnL1Lp7a31+7+f6cfaPKt1XqkM7vZ25x3J0d6nhMmR10PnXmNHxBwBNk8ViOanpNk+Wx+OR2+5WjCOm1mvWVVZW6rXXXtMzzzyjgQMHmp4bNmyY/v3vf6tPnz7Kycnxf8n5aL1795bH49EXX3zhn+rzaG3atNGhQ4dUXFzsL+6tXLnyhHHt3btX69at0/Tp03XBBRdIkhYvXmzap0+fPvrXv/6lffv2HbPr709/+pN69eqll156SZWVlbryyitP+Npo2Hx1Y6b6BACgaQjpr/yCggK53W4lJpovjicmJmrt2rXVHpOXl1ft/nl5edXuX15ervLycv/jwsJCSd6L7NV9Y64ufOcL9HnDFfkwIx9m5MMsGPlwe449vWagOFo45JBDle5KyfdyNsnR0iFHS/M6ch555HFV7SCL7RCr2A7mdVxcLpd2Z+9Wh4s6+Auhlb6vqUqyN7er4yUd1fGSjlWO87FGWdXqtFZqdVr1FzRCrcMFVYu6x+JyuZSdna0BAwYcd6rgY/HlztHCoT6j+6jP6D7yuD2yWLwFyqKdRfK4PaYpPJ3NnHLEeQu2R6/XeCItu3vXTR3sGuxd13N/mUr3lno7Q6PtatWtleJOifMXDgdMHqBLnrxEhdsK/VP5lu0vk+ExFN8jXnGnxPnXnIyIjJDhMVS8r1g5n+So++Xdg/JvCP8uNQzNHM2UmZyp3r170/EHAGhQ/vOf/2j//v26+eab1bx5c9NzV111lWbMmKGnn35al156qbp06aKrr75alZWVmjdvnv7+978rJSVFN9xwg2666SY9//zzSk1N1datW7V792798Y9/VFpamqKjo3Xffffpjjvu0NKlSzVr1qwTxtWyZUu1bt1a//znP9W2bVvl5uZWmUnpmmuu0eOPP65hw4Zp0qRJSkxM1Ndff60uXbrovPO8MymcdtppOuecc/T3v/9dN9100wm7BNHwtWsn/fnPK9SvXx81gMm/AABAkDX63/aTJk0yLabts2DBAkVH13waj5ORnZ0dlPOGK/JhRj7MyIcZ+TAjH2ZhnY8ISRWSfjr8czyHfz1v2b5F2l79Lo4Eh/737f8CFt7RSkpKTrxTiEydOlVPP/208vLylJqaqhdeeEH9+/evdt+LL75YX3xRdd3PIUOG6JNPPpEk3XjjjXr11VdNzw8aNEjz588PfPAnKcoepYzWGRrSd0ioQwEAwGTGjBnKyMioUvSTvIW/p556Sq1atdLbb7+tRx55RE888YTi4uJ04YVH1md/+eWXdd999+m2227T3r17dcopp+i+++6TJLVq1UpvvPGG7rnnHk2fPl2XXnqpHnroId1yyy3HjctqtWrOnDm644471KtXL3Xv3l3PP/+8Lr74Yv8+DodDCxYs0N13360hQ4aosrJS3bt310svvWQ6180336yvv/5aN910Ux0yhYaiZUvp0ku3aciQ3qEOBQAA1IOQFv7i4+Nls9mUn59v2p6fn6+kpKRqj0lKSjqp/ceNG2eaGrSwsFDJyckaOHCg4uKqn6avturakdHYkA8z8mFGPszIhxn5MCMfZsHOh292gIZm7ty5ysrK0rRp05SWlqYpU6Zo0KBBWrdunRISEqrs/9577/nX85G803+lpqbqD3/4g2m/wYMH65VXXvE/djqdwXsTAAA0Ah9//PExn+vfv79/BoU+ffocc5rMyMhITZ48WZMnT672+WHDhmnYsGGmbWPGjPHff+ihh/TQQw9VOS4jI0M///yzaduv1wvs2LGj3nnnHUmH1zwsLKxyfWTHjh3q3bu3zj777GrjAwAAQMMV0sKfw+FQv379lJOT4x/Qejwe5eTkKDMzs9pj0tPTlZOTo7vuusu/LTs7W+np6dXu73Q6q72AZbfbg3bxNJjnDkfkw4x8mJEPM/JhRj7MyIdZsPLRUHM8efJkjRkzxr9W0LRp0/TJJ59o5syZVabxklRl3Z45c+YoOjq6SuHP6XQe8wtUAACgaSkqKtKWLVv04osv6tFHHw11OAAAAKiFkE/1mZWVpRtuuEFnnXWW+vfvrylTpqi4uNh/UWvUqFFq3769Jk2aJEm68847ddFFF+mZZ57R5Zdfrjlz5ui7777TP//5z1C+DQAAgKCpqKjQ8uXLNW7cOP82q9WqjIwMLVmypEbnmDFjhq6++mrFxMSYti9atEgJCQlq2bKlLrnkEj366KNq3bp1tedg7eTQIR9m5MOMfJiRD7OGng+XyyXDMOTxeOTxVF0POtB83W++12zqfp2P22+/XXPmzNEVV1yhG2+8sUY58ng8MgxDLpdLNpvN9FxD/dwBAAA0ZiEv/I0YMUJ79uzR+PHjlZeXp759+2r+/PlKTEyUJOXm5spqtfr3P/fcczV79mw98MADuu+++9StWzd98MEH6tWrV6jeAgAAQFAVFBTI7Xb7x0c+iYmJWrt27QmPX7ZsmX788UfNmDHDtH3w4MG68sor1alTJ23cuFH33XefLrvsMi1ZsqTKhTuJtZMbAvJhRj7MyIcZ+TBrqPmIiIhQUlKSioqKTFNUB9uhQ4fq7bXCgS8fzz33nJ577jlJUnFxcY2OraioUGlpqb788ktVVlaanmvIaycDAAA0ViEv/ElSZmbmMaf2XLRoUZVtf/jDH6pMUwUAAIDqzZgxQ71791b//v1N26+++mr//d69e6tPnz7q0qWLFi1apEsvvbTKeVg7OXTIhxn5MCMfZuTDrKHno6ysTNu2bVNsbKwiIyOD/nqGYejQoUNq1qyZLBZL0F+voQtEPsrKyhQVFaULL7ywyp9hQ107GQAAoDFrEIU/AAAAHFt8fLxsNpvy8/NN2/Pz80+4Pl9xcbHmzJmjhx9++ISv07lzZ8XHx2vDhg3VFv5YOzn0yIcZ+TAjH2bkw6yh5sPtdstischqtZpm+wkW39SVvtds6gKRD6vVKovFUu1nrCF+5gAAABo7RrkAAAANnMPhUL9+/ZSTk+Pf5vF4lJOTo/T09OMe+/bbb6u8vFzXXXfdCV9n+/bt2rt3r9q2bVvnmAEAOBm+teYQfvizAwAAaFgo/AEAAISBrKwsTZ8+Xa+++qrWrFmjsWPHqri4WKNHj5YkjRo1SuPGjaty3IwZMzRs2DC1bt3atL2oqEj33HOPvvnmG23ZskU5OTm64oor1LVrVw0aNKhe3hMAAL41ZetzfT8Elm8dP7r7AAAAGgam+gQAAAgDI0aM0J49ezR+/Hjl5eWpb9++mj9/vhITEyVJubm5VaboWrdunRYvXqwFCxZUOZ/NZtOqVav06quv6sCBA2rXrp0GDhyoRx55pNrpPAEACIaIiAhFR0drz549stvtQZ9+0+PxqKKiQmVlZUz1qbrlwzAMlZSUaPfu3WrRooW/iAsAAIDQovAHAAAQJjIzM5WZmVntc4sWLaqyrXv37secfisqKkqfffZZIMMDAOCkWSwWtW3bVps3b9bWrVuD/nqGYai0tFRRUVGyWCxBf72GLhD5aNGixQnXHAYAAED9ofAHAAAAAABCxuFwqFu3bvUy3afL5dKXX36pCy+8kKkpVfd82O12Ov0AAAAaGAp/AAAAAAAgpKxWqyIjI4P+OjabTZWVlYqMjKTwJ/IBAADQGDGhPQAAAAAAAAAAANAIUPgDAAAAAAAAAAAAGgEKfwAAAAAAAAAAAEAj0OTW+DMMQ5JUWFgY8HO7XC6VlJSosLCQufFFPn6NfJiRDzPyYUY+zMiHWbDz4Rsj+MYMODbGVfWHfJiRDzPyYUY+zMiHGfkwY1zVcDCuqj/kw4x8mJEPM/JhRj7MyIdZQxpXNbnC36FDhyRJycnJIY4EAAA0ZIcOHVLz5s1DHUaDxrgKAADUBOOqE2NcBQAAaqIm4yqL0cS+duXxeLRz5041a9ZMFosloOcuLCxUcnKytm3bpri4uICeOxyRDzPyYUY+zMiHGfkwIx9mwc6HYRg6dOiQ2rVrJ6uVWdGPh3FV/SEfZuTDjHyYkQ8z8mFGPswYVzUcjKvqD/kwIx9m5MOMfJiRDzPyYdaQxlVNruPParWqQ4cOQX2NuLg4PuhHIR9m5MOMfJiRDzPyYUY+zIKZD76RXjOMq+of+TAjH2bkw4x8mJEPM/Jhxrgq9BhX1T/yYUY+zMiHGfkwIx9m5MOsIYyr+LoVAAAAAAAAAAAA0AhQ+AMAAAAAAAAAAAAaAQp/AeR0OjVhwgQ5nc5Qh9IgkA8z8mFGPszIhxn5MCMfZuSjaeDP2Yx8mJEPM/JhRj7MyIcZ+TAjH00Df85m5MOMfJiRDzPyYUY+zMiHWUPKh8UwDCPUQQAAAAAAAAAAAACoGzr+AAAAAAAAAAAAgEaAwh8AAAAAAAAAAADQCFD4AwAAAAAAAAAAABoBCn8BMnXqVKWkpCgyMlJpaWlatmxZqEOqFw899JAsFovpp0ePHv7ny8rKdPvtt6t169aKjY3VVVddpfz8/BBGHFhffvmlhg4dqnbt2sliseiDDz4wPW8YhsaPH6+2bdsqKipKGRkZ+uWXX0z77Nu3TyNHjlRcXJxatGihm2++WUVFRfX4LgLnRPm48cYbq3xeBg8ebNqnMeVj0qRJOvvss9WsWTMlJCRo2LBhWrdunWmfmvwdyc3N1eWXX67o6GglJCTonnvuUWVlZX2+lYCoST4uvvjiKp+RW2+91bRPY8nHyy+/rD59+iguLk5xcXFKT0/Xp59+6n++KX02pBPnoyl9NsC4inEV4yqJcdWvMa4yY1xlxrjKjHEVjsa4inEV4yrGVb/GuMqMcZUZ4yqzcB1XUfgLgLlz5yorK0sTJkzQihUrlJqaqkGDBmn37t2hDq1enH766dq1a5f/Z/Hixf7n/vKXv+jjjz/W22+/rS+++EI7d+7UlVdeGcJoA6u4uFipqamaOnVqtc8/9dRTev755zVt2jQtXbpUMTExGjRokMrKyvz7jBw5Uj/99JOys7P1n//8R19++aVuueWW+noLAXWifEjS4MGDTZ+Xf//736bnG1M+vvjiC91+++365ptvlJ2dLZfLpYEDB6q4uNi/z4n+jrjdbl1++eWqqKjQ119/rVdffVWzZs3S+PHjQ/GW6qQm+ZCkMWPGmD4jTz31lP+5xpSPDh066IknntDy5cv13Xff6ZJLLtEVV1yhn376SVLT+mxIJ86H1HQ+G00d4yrGVYyrvBhXmTGuMmNcZca4yoxxFXwYVzGuYlzlxbjKjHGVGeMqM8ZVZmE7rjJQZyR0uooAAA2oSURBVP379zduv/12/2O32220a9fOmDRpUgijqh8TJkwwUlNTq33uwIEDht1uN95++23/tjVr1hiSjCVLltRThPVHkvH+++/7H3s8HiMpKcl4+umn/dsOHDhgOJ1O49///rdhGIbx888/G5KMb7/91r/Pp59+algsFmPHjh31Fnsw/DofhmEYN9xwg3HFFVcc85jGnA/DMIzdu3cbkowvvvjCMIya/R2ZN2+eYbVajby8PP8+L7/8shEXF2eUl5fX7xsIsF/nwzAM46KLLjLuvPPOYx7TmPNhGIbRsmVL41//+leT/2z4+PJhGHw2mhLGVanVPse4inEV4yozxlVmjKuqYlxlxriqaWJclVrtc4yrGFcxrjJjXGXGuKoqxlVm4TCuouOvjioqKrR8+XJlZGT4t1mtVmVkZGjJkiUhjKz+/PLLL2rXrp06d+6skSNHKjc3V5K0fPlyuVwuU2569OihU045pUnkZvPmzcrLyzO9/+bNmystLc3//pcsWaIWLVrorLPO8u+TkZEhq9WqpUuX1nvM9WHRokVKSEhQ9+7dNXbsWO3du9f/XGPPx8GDByVJrVq1klSzvyNLlixR7969lZiY6N9n0KBBKiwsNH2zJBz9Oh8+b775puLj49WrVy+NGzdOJSUl/ucaaz7cbrfmzJmj4uJipaenN/nPxq/z4dMUPxtNDeMqxlXHwriqeoyrGFf5MK46gnGVGeOqpotxFeOqY2FcVT3GVYyrfBhXHcG4yiycxlURQTtzE1FQUCC32236g5OkxMRErV27NkRR1Z+0tDTNmjVL3bt3165duzRx4kRdcMEF+vHHH5WXlyeHw6EWLVqYjklMTFReXl5oAq5HvvdY3WfD91xeXp4SEhJMz0dERKhVq1aNMkeDBw/WlVdeqU6dOmnjxo267777dNlll2nJkiWy2WyNOh8ej0d33XWXzjvvPPXq1UuSavR3JC8vr9rPkO+5cFVdPiTp2muvVceOHdWuXTutWrVKf//737Vu3Tq99957khpfPlavXq309HSVlZUpNjZW77//vnr27KmVK1c2yc/GsfIhNb3PRlPFuIpx1bEwrqqKcRXjKh/GVV6Mq8wYV4FxFeOqY2FcVRXjKsZVPoyrvBhXmYXjuIrCH+rksssu89/v06eP0tLS1LFjR7311luKiooKYWRoiK6++mr//d69e6tPnz7q0qWLFi1apEsvvTSEkQXf7bffrh9//NG0pkBTdqx8HD0/fu/evdW2bVtdeuml2rhxo7p06VLfYQZd9+7dtXLlSh08eFDvvPOObrjhBn3xxRehDitkjpWPnj17NrnPBpomxlU4GYyrGFf5MK7yYlxlxrgKTR3jKpwMxlWMq3wYV3kxrjILx3EVU33WUXx8vGw2m/Lz803b8/PzlZSUFKKoQqdFixY69dRTtWHDBiUlJamiokIHDhww7dNUcuN7j8f7bCQlJVVZVLuyslL79u1rEjnq3Lmz4uPjtWHDBkmNNx+ZmZn6z3/+o//+97/q0KGDf3tN/o4kJSVV+xnyPReOjpWP6qSlpUmS6TPSmPLhcDjUtWtX9evXT5MmTVJqaqqee+65JvvZOFY+qtPYPxtNFeMqM8ZVRzCuOjHGVU3zdyfjqiMYV5kxrgLjKjPGVUcwrjoxxlVN83cn46ojGFeZheO4isJfHTkcDvXr1085OTn+bR6PRzk5OaZ5XpuKoqIibdy4UW3btlW/fv1kt9tNuVm3bp1yc3ObRG46deqkpKQk0/svLCzU0qVL/e8/PT1dBw4c0PLly/37LFy4UB6Px/+PRGO2fft27d27V23btpXU+PJhGIYyMzP1/vvva+HCherUqZPp+Zr8HUlPT9fq1atNA8zs7GzFxcX5W8rDxYnyUZ2VK1dKkukz0ljyUR2Px6Py8vIm99k4Fl8+qtPUPhtNBeMqM8ZVRzCuOjHGVU3rdyfjqhNjXGXGuKrpYVxlxrjqCMZVJ8a4qmn97mRcdWKMq8zCYlxloM7mzJljOJ1OY9asWcbPP/9s3HLLLUaLFi2MvLy8UIcWdHfffbexaNEiY/PmzcZXX31lZGRkGPHx8cbu3bsNwzCMW2+91TjllFOMhQsXGt99952Rnp5upKenhzjqwDl06JDx/fffG99//70hyZg8ebLx/fffG1u3bjUMwzCeeOIJo0WLFsaHH35orFq1yrjiiiuMTp06GaWlpf5zDB482DjjjDOMpUuXGosXLza6detmXHPNNaF6S3VyvHwcOnTI+Otf/2osWbLE2Lx5s/H5558bZ555ptGtWzejrKzMf47GlI+xY8cazZs3NxYtWmTs2rXL/1NSUuLf50R/RyorK41evXoZAwcONFauXGnMnz/faNOmjTFu3LhQvKU6OVE+NmzYYDz88MPGd999Z2zevNn48MMPjc6dOxsXXnih/xyNKR/33nuv8cUXXxibN282Vq1aZdx7772GxWIxFixYYBhG0/psGMbx89HUPhtNHeMqxlWMq7wYV5kxrjJjXGXGuMqMcRV8GFcxrmJc5cW4yoxxlRnjKjPGVWbhOq6i8BcgL7zwgnHKKacYDofD6N+/v/HNN9+EOqR6MWLECKNt27aGw+Ew2rdvb4wYMcLYsGGD//nS0lLjtttuM1q2bGlER0cbw4cPN3bt2hXCiAPrv//9ryGpys8NN9xgGIZheDwe48EHHzQSExMNp9NpXHrppca6detM59i7d69xzTXXGLGxsUZcXJwxevRo49ChQyF4N3V3vHyUlJQYAwcONNq0aWPY7XajY8eOxpgxY6r8h6Mx5aO6XEgyXnnlFf8+Nfk7smXLFuOyyy4zoqKijPj4eOPuu+82XC5XPb+bujtRPnJzc40LL7zQaNWqleF0Oo2uXbsa99xzj3Hw4EHTeRpLPm666SajY8eOhsPhMNq0aWNceuml/kGUYTStz4ZhHD8fTe2zAcZVjKsYVxkG46pfY1xlxrjKjHGVGeMqHI1xFeMqxlWMq36NcZUZ4yozxlVm4TqushiGYdS+XxAAAAAAAAAAAABAQ8AafwAAAAAAAAAAAEAjQOEPAAAAAAAAAAAAaAQo/AEAAAAAAAAAAACNAIU/AAAAAAAAAAAAoBGg8AcAAAAAAAAAAAA0AhT+AAAAAAAAAAAAgEaAwh8AAAAAAAAAAADQCFD4AwAAAAAAAAAAABoBCn8AUEsWi0UffPBBqMMAAAAIe4yrAAAAAoNxFQAKfwDC0o033iiLxVLlZ/DgwaEODQAAIKwwrgIAAAgMxlUAGoKIUAcAALU1ePBgvfLKK6ZtTqczRNEAAACEL8ZVAAAAgcG4CkCo0fEHIGw5nU4lJSWZflq2bCnJO63Byy+/rMsuu0xRUVHq3Lmz3nnnHdPxq1ev1iWXXKKoqCi1bt1at9xyi4qKikz7zJw5U6effrqcTqfatm2rzMxM0/MFBQUaPny4oqOj1a1bN3300UfBfdMAAABBwLgKAAAgMBhXAQg1Cn8AGq0HH3xQV111lX744QeNHDlSV199tdasWSNJKi4u1qBBg9SyZUt9++23evvtt/X555+bBkovv/yybr/9dt1yyy1avXq1PvroI3Xt2tX0GhMnTtQf//hHrVq1SkOGDNHIkSO1b9++en2fAAAAwca4CgAAIDAYVwEIOgMAwtANN9xg2Gw2IyYmxvTz2GOPGYZhGJKMW2+91XRMWlqaMXbsWMMwDOOf//yn0bJlS6OoqMj//CeffGJYrVYjLy/PMAzDaNeunXH//fcfMwZJxgMPPOB/XFRUZEgyPv3004C9TwAAgGBjXAUAABAYjKsANASs8QcgbP3mN7/Ryy+/bNrWqlUr//309HTTc+np6Vq5cqUkac2aNUpNTVVMTIz/+fPOO08ej0fr1q2TxWLRzp07demllx43hj59+vjvx8TEKC4uTrt3767tWwIAAAgJxlUAAACBwbgKQKhR+AMQtmJiYqpMZRAoUVFRNdrPbrebHlssFnk8nmCEBAAAEDSMqwAAAAKDcRX+f3t3rJJsFIcB/DGahDYp3NzE5ty6AbegtgjXEKSlPbuBvALHSGhoraFRiLY2LyFoDEEnvy2IbzWtw++3veeFl//ZHng474FNc8cfUKyXl5f/nlutVpKk1Wrl7e0ts9ns6/1kMsnW1laazWZ2dnbSaDTy/Py81pkBAH4juQoAYDXkKuCnOfEH/FmLxSLv7+/f1ra3t1Or1ZIk9/f3OTg4yOHhYW5vb/P6+prRaJQkOT09zdXVVbrdbgaDQT4+PtLv93N2dpa9vb0kyWAwyPn5eXZ3d9PpdPL5+ZnJZJJ+v7/ejQIA/DC5CgBgNeQqYNMUf8Cf9fj4mHq9/m2t2WxmOp0mSa6vrzMej9Pr9VKv13N3d5f9/f0kSbVazdPTUy4uLtJut1OtVnN8fJybm5uvb3W73czn8wyHw1xeXqZWq+Xk5GR9GwQAWBO5CgBgNeQqYNMqy+VyuekhAFatUqnk4eEhR0dHmx4FAOBPk6sAAFZDrgLWwR1/AAAAAAAAUADFHwAAAAAAABTArz4BAAAAAACgAE78AQAAAAAAQAEUfwAAAAAAAFAAxR8AAAAAAAAUQPEHAAAAAAAABVD8AQAAAAAAQAEUfwAAAAAAAFAAxR8AAAAAAAAUQPEHAAAAAAAABVD8AQAAAAAAQAH+AU29F5DLOp+GAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 0 Axes>"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "---\n",
        " # **Loading the Trained Crack Segmentation Model**\n",
        "\n",
        "---\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "ww6xw6vmP6c-"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Saving in the google drive**"
      ],
      "metadata": {
        "id": "zYZQ8dUQeLKv"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Re-import your libraries (after runtime reset)\n",
        "import segmentation_models_pytorch as smp\n",
        "import torch\n",
        "import albumentations as A\n",
        "from albumentations.pytorch import ToTensorV2\n",
        "\n",
        "# Define the device (GPU or CPU)\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "\n",
        "# Define the same model architecture\n",
        "model = smp.Unet(\n",
        "    encoder_name=\"resnet34\",\n",
        "    encoder_weights=\"imagenet\",  # Use ImageNet weights for the encoder\n",
        "    in_channels=3,\n",
        "    classes=6,\n",
        ")\n",
        "\n",
        "# Load the trained model weights\n",
        "model.load_state_dict(torch.load(\"/content/drive/MyDrive/CrackDetection/crack_segmentation_model.pth\", map_location=device))\n",
        "\n",
        "# Move the model to the appropriate device (GPU/CPU)\n",
        "model = model.to(device)\n",
        "\n",
        "# Set the model to evaluation mode\n",
        "model.eval()\n",
        "\n",
        "# Define the transform (same as training)\n",
        "transform = A.Compose([\n",
        "    A.Resize(256, 256),\n",
        "    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),  # Normalize like during training\n",
        "    ToTensorV2()\n",
        "])\n",
        "\n",
        "print(\"✅ Model loaded and ready for prediction!\")\n"
      ],
      "metadata": {
        "id": "yBC9N3SCeKUv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Prediction of Crack on any Random Images**"
      ],
      "metadata": {
        "id": "7s2ZIREeJKT0"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import cv2\n",
        "import torch\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Load saved model\n",
        "model = smp.Unet(\n",
        "    encoder_name=\"resnet34\",\n",
        "    encoder_weights=None,\n",
        "    in_channels=3,\n",
        "    classes=6,\n",
        ")\n",
        "model.load_state_dict(torch.load(\"/content/drive/MyDrive/CrackDetection/crack_segmentation_model.pth\", map_location=device))\n",
        "model = model.to(device)\n",
        "model.eval()\n",
        "\n",
        "# Class labels\n",
        "class_names = {\n",
        "    0: \"Background\",\n",
        "    1: \"Compression Crack\",\n",
        "    2: \"Rebar Detachment\",\n",
        "    3: \"Shear Type - 01\",\n",
        "    4: \"Shear Type - 02\",\n",
        "    5: \"Tension Crack\"\n",
        "}\n",
        "\n",
        "# Prediction & visualization\n",
        "def predict_image(path, transform):\n",
        "    image = cv2.imread(path)\n",
        "    if image is None:\n",
        "        print(f\"❌ Could not read image: {path}\")\n",
        "        return\n",
        "    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\n",
        "\n",
        "    # Transform\n",
        "    augmented = transform(image=image)\n",
        "    input_tensor = augmented['image'].unsqueeze(0).to(device)\n",
        "\n",
        "    # Predict\n",
        "    with torch.no_grad():\n",
        "        output = model(input_tensor)\n",
        "        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()\n",
        "\n",
        "    # Report detected classes\n",
        "    detected = np.unique(pred_mask)\n",
        "    print(f\"\\n🔍 Predicting: {os.path.basename(path)}\")\n",
        "    print(\"🧠 Crack Types Detected:\")\n",
        "    for cls_id in detected:\n",
        "        print(f\"  ➤ Class {cls_id}: {class_names.get(cls_id, 'Unknown')}\")\n",
        "\n",
        "    # Visualize\n",
        "    plt.figure(figsize=(10, 4))\n",
        "    plt.subplot(1, 2, 1)\n",
        "    plt.imshow(image)\n",
        "    plt.title(\"Input Image\")\n",
        "    plt.axis('off')\n",
        "\n",
        "    plt.subplot(1, 2, 2)\n",
        "    plt.imshow(pred_mask, cmap='jet', vmin=0, vmax=5)\n",
        "    plt.title(\"Predicted Crack Mask\")\n",
        "    plt.axis('off')\n",
        "    plt.show()\n",
        "\n",
        "# Run prediction on all images in the folder\n",
        "folder_path = '/content/drive/MyDrive/B-1300-5.56'\n",
        "image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]\n",
        "\n",
        "print(f\"🖼 Found {len(image_files)} images\")\n",
        "\n",
        "for img_file in image_files:\n",
        "    predict_image(os.path.join(folder_path, img_file), transform)\n"
      ],
      "metadata": {
        "id": "BHGxdxi7Xhbj"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
