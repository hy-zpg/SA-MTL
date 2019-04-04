## Data preparation:
   * storing datasets
   * prepare images-labels pairs
   * generating csv file for generator

## Utils:
   stl: prepare data generator for single-task learning
   confusion: prepare data generator for multi-task learning with joint training method
   alternative: prepare data generator for single-task learning with alternative triaing method

## Comparison between SA-MTL and other multi-task learning methods, as well as single-task learning
### Alternative-MTL:
E,P: emotion, pose

Runing script:
``` bash
python ALTERNATIVE_EP.py {other similar function, ALTERNATIVE_EA.py, ALTERNATIVE_EGA.py} 
--dataset_emotion=expw --dataset_pose=aflw --epoch=64 --model=vggFace  --batch_size=32 --is_augmentation=False --is_dropout=False --is_bn=False --weights_decay=0 --is_freezing=False --no_freezing_epoch=0 --P_loss_weights=1 --E_loss_weights=1 --is_naive=False --is_distilled=False distill_t=2 --is_pesudo=True --is_interpolation=False --interpolation_weights=0 --selection_threshold=0.8 --is_pesudo_confidence=True --is_pesudo_density=True --density_t=0.6 --is_pesudo_distribution=True --cluster_k=3 
```

Explanation:
* Dataset setting: --dataset_emotion [Expw, Fer+, Fer2013, SFEW], --dataset_pose [AFLW]
* Model setting: --model [VGGFace, mini_xception]
* Training tricks: --is_augmentation=False --is_dropout=False --is_bn=False --weights_decay=0 --is_freezing=False --no_freezing_epoch=0 [data augmentation, dropout, batchnormalization, weights decay, two-stage training]
* Multi-task hyperparameter: --P_loss_weights=1 --E_loss_weights=1 [weights of different tasks]
* Core methods: 
   $ --is_naive=True: not generate pseudo labels

   $ --is_distilled=True, --distill_t: only leverage distilled knowledge to preserve information, distill_t is the temperature

   $ --is_pesudo=True: generateb pseudo labels to augment infromation

   $ --is_interpolation=True: generate interpolated labels combining distilled knowledge and pseudo labels with selected weights, otherwise adopt hard selection method to select high-confidence pseudo labels to augment information, while distilled knowledge is adopt to preserver information for those low-confidence pseudo labels 

   $ --selection_threshold: select pseudo labels for hard selection method

   $ --is_pesudo_confidence=True: leveage confidence score to generate pseudo labels weights

   $ --is_pesudo_density=True, --density_t=0.6: leverage local density to generate pseudo labels weights

   $ --is_pesudo_distribution=True, --cluster_k=3 : leverage MMD and EMD to generate pseudo labels weights

Comparison methods:
* DML-LWF: 
``` bash
python ALTERNATIVE_EP.py --dataset_emotion=expw --dataset_pose=aflw --epoch=64 --model=vggFace  --batch_size=32 --is_augmentation=False --is_dropout=False --is_bn=False --weights_decay=0 --is_freezing=False --no_freezing_epoch=0 --P_loss_weights=1 --E_loss_weights=1 --is_naive=False --is_distilled=True distill_t=1 --is_pesudo=False--is_interpolation=False --interpolation_weights=0 --selection_threshold=0 --is_pesudo_confidence=False --is_pesudo_density=False --density_t=0 --is_pesudo_distribution=False --cluster_k=0 
```

* distilling tricks:
``` bash
python ALTERNATIVE_EP.py --dataset_emotion=expw --dataset_pose=aflw --epoch=64 --model=vggFace  --batch_size=32 --is_augmentation=False --is_dropout=False --is_bn=False --weights_decay=0 --is_freezing=False --no_freezing_epoch=0 --P_loss_weights=1 --E_loss_weights=1 --is_naive=False --is_distilled=True distill_t=2 --is_pesudo=False--is_interpolation=False --interpolation_weights=0 --selection_threshold=0 --is_pesudo_confidence=False --is_pesudo_density=False --density_t=0 --is_pesudo_distribution=False --cluster_k=0
```

* SA-MTL: 
``` bash
python ALTERNATIVE_EP.py --dataset_emotion=expw --dataset_pose=aflw --epoch=64 --model=vggFace  --batch_size=32 --is_augmentation=False --is_dropout=False --is_bn=False --weights_decay=0 --is_freezing=False --no_freezing_epoch=0 --P_loss_weights=1 --E_loss_weights=1 --is_naive=False --is_distilled=False distill_t=2 --is_pesudo=True --is_interpolation=True --interpolation_weights=0 --selection_threshold=0 --is_pesudo_confidence=True --is_pesudo_density=True --density_t=0.6 --is_pesudo_distribution=True --cluster_k=4
```

### Joint-MTL
The papameters is similar to above mentioned

Comparison method: 
1. pseudo labels method:

* All-in-one: 
``` bash
python CONFUSION_EP_multitask.py --dataset_emotion=SFEW --dataset_pose=aflw --epoch=64 --model=vggFace  --batch_size=64 --is_augmentation=False --is_dropout=False --is_bn=False --weights_decay=0 --is_freezing=False --no_freezing_epoch=0 --P_loss_weights=1 --E_loss_weights=1 --is_naive=True --is_distilled=False --is_pesudo=False --is_interpolation=False --interpolation_weights=0 --selection_threshold=0.8 --is_pesudo_confidence=False --is_pesudo_density=False --is_pesudo_distribution=False --cluster_k=0
```

* LEL-LTN: 
``` bash
python CONFUSION_EP_multitask.py --dataset_emotion=SFEW --dataset_pose=aflw --epoch=64 --model=vggFace  --batch_size=64 --is_augmentation=False --is_dropout=False --is_bn=False --weights_decay=0 --is_freezing=False --no_freezing_epoch=0 --P_loss_weights=1 --E_loss_weights=1 --is_naive=False --is_distilled=False --is_pesudo=True --is_interpolation=False --interpolation_weights=0 --selection_threshold=0 --is_pesudo_confidence=False --is_pesudo_density=False --is_pesudo_distribution=False --cluster_k=0
```

* DCN-AP: 
``` bash
python CONFUSION_EP_multitask.py --dataset_emotion=SFEW --dataset_pose=aflw --epoch=64 --model=vggFace  --batch_size=64 --is_augmentation=False --is_dropout=False --is_bn=False --weights_decay=0 --is_freezing=False --no_freezing_epoch=0 --P_loss_weights=1 --E_loss_weights=1 --is_naive=False --is_distilled=False --is_pesudo=True --is_interpolation=False --interpolation_weights=0 --selection_threshold=0 --is_pesudo_confidence=False --is_pesudo_density=False --is_pesudo_distribution=False --cluster_k=0 --is_MRF=True
```


2. manifold methods:
model: select four outputs including features and predicted labels, predicted labels for cross-entropy loss while features for regularization(such as l_{2,1} term, trace norm term, nuclear norm, laplacian norm), setting the parameter is_manifold as True.

* SFSMR: 
``` bash
python CONFUSION_EP_multitask_manifold.py --dataset_emotion=SFEW --dataset_pose=aflw --epoch=64 --model=vggFace  --batch_size=64 --is_augmentation=False --is_dropout=False --is_bn=False --weights_decay=0 --is_freezing=False --no_freezing_epoch=0 --P_loss_weights=1 --E_loss_weights=1 --is_naive=True --is_distilled=False --is_pesudo=False --is_interpolation=False --interpolation_weights=0 --selection_threshold=0 --is_pesudo_confidence=False --is_pesudo_density=False --is_pesudo_distribution=False --cluster_k=0 --is_lnorm=True --is_trace_norm=True
```

* SLRM:
``` bash 
python CONFUSION_EP_multitask_manifold.py --dataset_emotion=SFEW --dataset_pose=aflw --epoch=64 --model=vggFace  --batch_size=64 --is_augmentation=False --is_dropout=False --is_bn=False --weights_decay=0 --is_freezing=False --no_freezing_epoch=0 --P_loss_weights=1 --E_loss_weights=1 --is_naive=True --is_distilled=False --is_pesudo=False --is_interpolation=False --interpolation_weights=0 --selection_threshold=0 --is_pesudo_confidence=False --is_pesudo_density=False --is_pesudo_distribution=False --cluster_k=0 --is_lnorm=True --is_nuclear_norm=True --is_trace_norm=True
```



### STL
Runing script:
* Emotion: 
``` bash
python STL_general_train.py --dataset=SFEW[expw,ferplus] --model=vggFace --epoch=64 --batch_size=32 --is_augmentation=False --is_dropout=False --is_bn=False --weights_decay=0 --is_freezing=False --no_freezing_epoch=0 --task_type=0
```

* pose: 
``` bash
python STL_general_train.py --dataset=aflw --model=vggFace --epoch=64 --batch_size=32 --is_augmentation=False --is_dropout=False --is_bn=False --weights_decay=0 --is_freezing=False --no_freezing_epoch=0 --task_type=10
```

## Demo for emotion, pose, gender and age recognition
* image: 
``` bash
python final_image_demo.py
``` 
* video: 
``` bash
python final_video_demo.py
``` 



