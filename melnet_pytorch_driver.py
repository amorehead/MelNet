#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install jamo')
get_ipython().system('pip install audiosegment')
get_ipython().system('pip install unidecode')
get_ipython().system('pip install nltk')
get_ipython().system('pip install importlib')
# !pip install -r /content/drive/My\ Drive/Tremendous\ Storytime/MelNet/requirements.txt # Google Colab
# !pip install -r ~/Repositories/MelNet/requirements.txt # Local machine
get_ipython().system('pip install -r ../input/melnet/requirements.txt # Kaggle')

# In[ ]:


# !python3 /content/drive/My\ Drive/Tremendous\ Storytime/MelNet/melnet_pytorch.py -c /content/drive/My\ Drive/Tremendous\ Storytime/MelNet/config/trump.yaml -n tremendous_storytime -t 1 -b 1 -s True # Google Colab
# !python3 ~/Repositories/MelNet/melnet_pytorch.py -c ~/Repositories/MelNet/config/trump.yaml -n tremendous_storytime -t 1 -b 1 -s True # Local machine
get_ipython().system(
    'python3 ../input/melnet/melnet_pytorch.py -c ../input/melnet/config/trump.yaml -n tremendous_storytime -t 1 -b 1 -s True # Kaggle')

# In[ ]:
