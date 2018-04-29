import os
import os.path
import numpy as np
import pickle

master_path = os.environ['NHANES_PROJECT_ROOT']
data_path = os.path.join(master_path, 'data')
out_path = os.path.join(data_path, 'all')

# lab
lab_path = os.path.join(data_path, 'lab')
lab_data = np.load(os.path.join(lab_path, 'quantized_dense_labdata_2013-2014.npy'))
lab_labels = np.load(os.path.join(lab_path, 'quantized_dense_labdata_info_2013-2014.npy'))

#demographics
demo_path = os.path.join(data_path, 'demo')
demo_data = np.load(os.path.join(demo_path, 'demographic_data_2013-1014.npy'))
demo_labels = np.load(os.path.join(demo_path, 'demographic_column_headers_2013-2014.npy'))

#dietary
diet_path = os.path.join(data_path, 'diet')
diet_data = np.load(os.path.join(diet_path, 'dietary_data.npy'))
diet_labels = np.load(os.path.join(diet_path, 'dietary_labels.npy'))
diet_mask = np.load(os.path.join(diet_path, 'dietary_miss_mask.npy'))


# set diet missing values to 9
diet_data[diet_mask] = 9

#aggregate data
data = np.hstack((demo_data, lab_data, diet_data)).astype(np.float)

#process and aggergate labels
demo_labels_json = np.array([{'name':label} for label in demo_labels], dtype=object)
diet_labels_json = np.array([{'name':label} for label in diet_labels], dtype=object)
labels = np.hstack((demo_labels_json, lab_labels, diet_labels_json))

#make info
info = {'offsets':
            {'demo':0,
            'lab': demo_labels.size,
            'diet': demo_labels.size + lab_labels.size
            },
        'labels':
            {'demo':
                {'min': np.min(demo_data),
                'max': np.max(demo_data)
                },
            'lab':
                {'min': np.min(lab_data),
                'max': np.max(lab_data[lab_data != 9])
                },
            'diet':
                {'min': np.min(diet_data),
                'max': np.max(diet_data)
                }
            }
       }
       
#save all
if not os.path.exists(out_path):
    os.makedirs(out_path)

np.save(os.path.join(out_path, 'data.npy'), data)
np.save(os.path.join(out_path, 'labels.npy'), labels)
pickle.dump(info, open(os.path.join(out_path, 'info.pkl'), "wb" ))
