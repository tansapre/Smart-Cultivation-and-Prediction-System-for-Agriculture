"""
# -*- coding: utf-8 -*-


Created on Sat May  8 13:40:01 2021

@author: 91758

"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import cv2
import base64
import io
import torch
import random
import operator
import SessionState
import requests
import webbrowser

from PIL import Image
from skimage import io

from google.cloud import firestore

from utils.disease import disease_dic
from utils.model import ResNet9
from utils.fertilizer import fertilizer_dic

from classify import predict
from torchvision import transforms
from flask import Flask, render_template, request, Markup

from tensorflow.keras.models import Sequential
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.models import load_model
global graph, model, output_list


fertilizer_dic = {
        'NHigh': """The N value of soil is high and might give rise to weeds.

         Please consider the following suggestions:

        1.  Manure  – adding manure is one of the simplest ways to amend your soil with nitrogen. Be careful as there are various types of manures with varying degrees of nitrogen.

        2. Coffee grinds  – use your morning addiction to feed your gardening habit! Coffee grinds are considered a green compost material which is rich in nitrogen. 
                            Once the grounds break down, your soil will be fed with delicious, delicious nitrogen. 
                            An added benefit to including coffee grounds to your soil is while it will compost, it will also help provide increased drainage to your soil.

        3. Plant nitrogen fixing plants – planting vegetables that are in Fabaceae family like peas, beans and soybeans have the ability to increase nitrogen in your soil

        4. Plant ‘green manure’ crops like cabbage, corn and brocolli

        5. Use mulch (wet grass) while growing crops - Mulch can also include sawdust and scrap soft woods""",

        'Nlow': """The N value of your soil is low.

         Please consider the following suggestions:

        1. Add sawdust or fine woodchips to your soil – the carbon in the sawdust/woodchips love nitrogen and will help absorb and soak up and excess nitrogen.

        2. Plant heavy nitrogen feeding plants – tomatoes, corn, broccoli, cabbage and spinach are examples of plants that thrive off nitrogen and will suck the nitrogen dry.

        3. Water – soaking your soil with water will help leach the nitrogen deeper into your soil, effectively leaving less for your plants to use.

        4. Sugar – In limited studies, it was shown that adding sugar to your soil can help potentially reduce the amount of nitrogen is your soil. 
                   Sugar is partially composed of carbon, an element which attracts and soaks up the nitrogen in the soil. 
                   This is similar concept to adding sawdust/woodchips which are high in carbon content.

        5. Add composted manure to the soil.

        6. Plant Nitrogen fixing plants like peas or beans.

        7. Use NPK fertilizers with high N value.

        8. Do nothing – It may seem counter-intuitive, but if you already have plants that are producing lots of foliage, 
                        it may be best to let them continue to absorb all the nitrogen to amend the soil for your next crops.""",

        'PHigh': """The P value of your soil is high.

         Please consider the following suggestions:

        1. Avoid adding manure – manure contains many key nutrients for your soil but typically including high levels of phosphorous. 
                                 Limiting the addition of manure will help reduce phosphorus being added.

        2. Use only phosphorus-free fertilizer – if you can limit the amount of phosphorous added to your soil, 
                                                 you can let the plants use the existing phosphorus while still providing other key nutrients such as Nitrogen and Potassium.
                                                 Find a fertilizer with numbers such as 10-0-10, where the zero represents no phosphorous.

        3. Water your soil – soaking your soil liberally will aid in driving phosphorous out of the soil. This is recommended as a last ditch effort.

        4. Plant nitrogen fixing vegetables to increase nitrogen without increasing phosphorous (like beans and peas).

        5. Use crop rotations to decrease high phosphorous levels""",

        'Plow': """The P value of your soil is low.

         Please consider the following suggestions:

        1. Bone meal – a fast acting source that is made from ground animal bones which is rich in phosphorous.

        2. Rock phosphate – a slower acting source where the soil needs to convert the rock phosphate into phosphorous that the plants can use.

        3. Phosphorus Fertilizers – applying a fertilizer with a high phosphorous content in the NPK ratio (example: 10-20-10, 20 being phosphorous percentage).

        4. Organic compost – adding quality organic compost to your soil will help increase phosphorous content.

        5. Manure – as with compost, manure can be an excellent source of phosphorous for your plants.

        6. Clay soil – introducing clay particles into your soil can help retain & fix phosphorus deficiencies.

        7. Ensure proper soil pH – having a pH in the 6.0 to 7.0 range has been scientifically proven to have the optimal phosphorus uptake in plants.

        8. If soil pH is low, add lime or potassium carbonate to the soil as fertilizers. 
        Pure calcium carbonate is very effective in increasing the pH value of the soil.

        9. If pH is high, addition of appreciable amount of organic matter will help acidify the soil. 
           Application of acidifying fertilizers, such as ammonium sulfate, can help lower soil pH""",

        'KHigh': """The K value of your soil is high.

         Please consider the following suggestions:

        1. Loosen the soil deeply with a shovel, and water thoroughly to dissolve water-soluble potassium. 
           Allow the soil to fully dry, and repeat digging and watering the soil two or three more times.

        2. Sift through the soil, and remove as many rocks as possible, using a soil sifter. 
           Minerals occurring in rocks such as mica and feldspar slowly release potassium into the soil slowly through weathering.

        3. Stop applying potassium-rich commercial fertilizer. Apply only commercial fertilizer that has a '0' in the final number field.  
           Commercial fertilizers use a three number system for measuring levels of nitrogen, phosphorous and potassium. The last number stands for potassium. 
           Another option is to stop using commercial fertilizers all together and to begin using only organic matter to enrich the soil.

        4. Mix crushed eggshells, crushed seashells, wood ash or soft rock phosphate to the soil to add calcium. 
        Mix in up to 10 percent of organic compost to help amend and balance the soil.

        5. Use NPK fertilizers with low K levels and organic fertilizers since they have low NPK values.

        6. Grow a cover crop of legumes that will fix nitrogen in the soil. This practice will meet the soil’s needs for nitrogen without increasing phosphorus or potassium.
        """,

        'Klow': """The K value of your soil is low.

        Please consider the following suggestions:

        1. Mix in muricate of potash or sulphate of potash
        2. Try kelp meal or seaweed
        3. Try Sul-Po-Mag
        4. Bury banana peels an inch below the soils surface
        5. Use Potash fertilizers since they contain high values potassium
        """
    }

disease_dic = {
    'Apple___Apple_scab': """ Crop: Apple  Disease: Apple Scab

          Cause of disease:
          
          1. Apple scab overwinters primarily in fallen leaves and in the soil. Disease development is favored by wet, cool weather that generally occurs in spring and early summer.
          
          2. Fungal spores are carried by wind, rain or splashing water from the ground to flowers, leaves or fruit.
             During damp or rainy periods, newly opening apple leaves are extremely susceptible to infection.
             The longer the leaves remain wet, the more severe the infection will be.
             Apple scab spreads rapidly between 55-75 degrees Fahrenheit.

          How to prevent/cure the disease:
          
          1. Choose resistant varieties when possible.
          
          2. Rake under trees and destroy infected leaves to reduce the number of fungal spores available to start the disease cycle over again next spring.
          
          3. Water in the evening or early morning hours (avoid overhead irrigation) to give the leaves time to dry out before infection can occur.
          
          4. Spread a 3- to 6-inch layer of compost under trees, keeping it away from the trunk, to cover soil and prevent splash dispersal of the fungal spores.""",

    'Apple___Black_rot': """ Crop: Apple  Disease: Black Rot

          Cause of disease:
          
          1. Black rot is caused by the fungus Diplodia seriata (syn Botryosphaeria obtusa).
             The fungus can infect dead tissue as well as living trunks, branches, leaves and fruits.
             In wet weather, spores are released from these infections and spread by wind or splashing water.
             The fungus infects leaves and fruit through natural openings or minor wounds.

          How to prevent/cure the disease:
          
          1. Prune out dead or diseased branches.
          
          2. Prune out dead or diseased branches.
          
          3. Remove infected plant material from the area.

          4. Remove infected plant material from the area.

          5. Be sure to remove the stumps of any apple trees you cut down. Dead stumps can be a source of spores.""",

    'Apple___Cedar_apple_rust': """ Crop: Apple  Disease: Cedar Apple Rust 

          Cause of disease:

          1. Cedar apple rust (Gymnosporangium juniperi-virginianae) is a fungal disease that depends on two species to spread and develop.
             It spends a portion of its two-year life cycle on Eastern red cedar (Juniperus virginiana).
             The pathogen’s spores develop in late fall on the juniper as a reddish brown gall on young branches of the trees.

         How to prevent/cure the disease:

         1. Since the juniper galls are the source of the spores that infect the apple trees, cutting them is a sound strategy if there aren’t too many of them.

         2. While the spores can travel for miles, most of the ones that could infect your tree are within a few hundred feet.

         3. The best way to do this is to prune the branches about 4-6 inches below the galls.""",

    'Apple___healthy': """ Crop: Apple  Disease: No disease 

         Don't worry. Your crop is healthy. Keep it up !!!""",


    'Blueberry___healthy': """ Crop: Blueberry  Disease: No disease 

         Don't worry. Your crop is healthy. Keep it up !!!""",

    'Cherry_(including_sour)___Powdery_mildew': """ Crop: Cherry  Disease: Powdery Mildew

         Cause of disease:
         
         1. Podosphaera clandestina, a fungus that most commonly infects young, expanding leaves but can also be found on buds, fruit and fruit stems.
            It overwinters as small, round, black bodies (chasmothecia) on dead leaves, on the orchard floor, or in tree crotches.
            Colonies produce more (asexual) spores generally around shuck fall and continue the disease cycle.

        How to prevent/cure the disease:
        
         1. Remove and destroy sucker shoots.

         2. Keep irrigation water off developing fruit and leaves by using irrigation that does not wet the leaves.
            Also, keep irrigation sets as short as possible.

         3. Follow cultural practices that promote good air circulation, such as pruning, and moderate shoot growth through judicious nitrogen management.""",

    'Cherry_(including_sour)___healthy': """ Crop: Cherry  Disease: No disease

         Don't worry. Your crop is healthy. Keep it up !!!""",


    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': """ Crop: Corn  Disease: Grey Leaf Spot

        Cause of disease:
        
        1. Gray leaf spot lesions on corn leaves hinder photosynthetic activity, reducing carbohydrates allocated towards grain fill.
           The extent to which gray leaf spot damages crop yields can be estimated based on the extent to which leaves are infected relative to grainfill.
           Damage can be more severe when developing lesions progress past the ear leaf around pollination time.
           Because a decrease in functioning leaf area limits photosynthates dedicated towards grainfill, the plant might mobilize more carbohydrates from the stalk to fill kernels.

        How to prevent/cure the disease:

        1. In order to best prevent and manage corn grey leaf spot, the overall approach is to reduce the rate of disease growth and expansion.

        2. This is done by limiting the amount of secondary disease cycles and protecting leaf area from damage until after corn grain formation.

        3. High risk factors for grey leaf spot in corn:
           a. Susceptible hybrid
           b. Continuous corn
           c. Late planting date
           d. Minimum tillage systems
           e. Field history of severe disease
           f. Early disease activity (before tasseling)
           g. Irrigation
           h. Favorable weather forecast for disease.""",


    'Corn_(maize)___Common_rust_': """ Crop: Corn(maize)  Disease: Common Rust

        Cause of disease:
        
        1. Common corn rust, caused by the fungus Puccinia sorghi, is the most frequently occurring of the two primary rust diseases of corn in the U.S., but it rarely causes significant yield losses in Ohio field (dent) corn.
        Occasionally field corn, particularly in the southern half of the state, does become severely affected when weather conditions favor the development and spread of rust fungus

        How to prevent/cure the disease:

        1. Although rust is frequently found on corn in Ohio, very rarely has there been a need for fungicide applications.
           This is due to the fact that there are highly resistant field corn hybrids available and most possess some degree of resistance.

        2. However, popcorn and sweet corn can be quite susceptible. In seasons where considerable rust is present on the lower leaves prior to silking and the weather is unseasonably cool and wet, an early fungicide application may be necessary for effective disease control. Numerous fungicides are available for rust control. """,


    'Corn_(maize)___Northern_Leaf_Blight': """ Crop: Corn(maize)  Disease: Northern Leaf Blight

        Cause of disease:
        
        1. Northern corn leaf blight (NCLB) is a foliar disease of corn (maize) caused by Exserohilum turcicum, the anamorph of the ascomycete Setosphaeria turcica.
           With its characteristic cigar-shaped lesions, this disease can cause significant yield loss in susceptible corn hybrids.

        How to prevent/cure the disease:

        1. Management of NCLB can be achieved primarily by using hybrids with resistance, but because resistance may not be complete or may fail, it is advantageous to utilize an integrated approach with different cropping practices and fungicides.

        2. Scouting fields and monitoring local conditions is vital to control this disease.""",


    'Grape___Black_rot': """ Crop: Grape  Disease: Black Rot

        Cause of disease:

        1. The black rot fungus overwinters in canes, tendrils, and leaves on the grape vine and on the ground.
           Mummified berries on the ground or those that are still clinging to the vines become the major infection source the following spring.

        2. During rain, microscopic spores (ascospores) are shot out of numerous, black fruiting bodies (perithecia) and are carried by air currents to young, expanding leaves.
           In the presence of moisture, these spores germinate in 36 to 48 hours and eventually penetrate the leaves and fruit stems.

        3. The infection becomes visible after 8 to 25 days. When the weather is wet, spores can be released the entire spring and summer providing continuous infection.

        How to prevent/cure the disease:

        1. Space vines properly and choose a planting site where the vines will be exposed to full sun and good air circulation.
           Keep the vines off the ground and insure they are properly tied, limiting the amount of time the vines remain wet thus reducing infection.

        2. Keep the fruit planting and surrounding areas free of weeds and tall grass. This practice will promote lower relative humidity and rapid drying of vines and thereby limit fungal infection.

        3. Use protective fungicide sprays. Pesticides registered to protect the developing new growth include copper, captan, ferbam, mancozeb, maneb, triadimefon, and ziram. Important spraying times are as new shoots are 2 to 4 inches long, and again when they are 10 to 15 inches long, just before bloom, just after bloom, and when the fruit has set.""",

    'Corn_(maize)___healthy': """ Crop: Corn(maize)  Disease: No disease 

        Don't worry. Your crop is healthy. Keep it up !!!""",


    'Grape___Esca_(Black_Measles)': """ Crop: Grape  Disease: Black Measles

        Cause of disease:

        1. Black Measles is caused by a complex of fungi that includes several species of Phaeoacremonium, primarily by P. aleophilum (currently known by the name of its sexual stage, Togninia minima), and by Phaeomoniella chlamydospora.

        2. The overwintering structures that produce spores (perithecia or pycnidia, depending on the pathogen) are embedded in diseased woody parts of vines.

        3. During fall to spring rainfall, spores are released and wounds made by dormant pruning provide infection sites.

        4. Wounds may remain susceptible to infection for several weeks after pruning with susceptibility declining over time.

        How to prevent/cure the disease:

        1. Post-infection practices (sanitation and vine surgery) for use in diseased, mature vineyards are not as effective and are far more costly than adopting preventative practices (delayed pruning, double pruning, and applications of pruning-wound protectants) in young vineyards.

        2. Sanitation and vine surgery may help maintain yields. In spring, look for dead spurs or for stunted shoots.
           Later in summer, when there is a reduced chance of rainfall, practice good sanitation by cutting off these cankered portions of the vine beyond the canker, to where wood appears healthy. Then remove diseased, woody debris from the vineyard and destroy it.

        3. The fungicides labeled as pruning-wound protectants, consider using alternative materials, such as a wound sealant with 5 percent boric acid in acrylic paint (Tech-Gro B-Lock), which is effective against Eutypa dieback and Esca, or an essential oil (Safecoat VitiSeal).""",

    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': """ Crop: Grape  Disease: Leaf Blight

        Cause of disease:

        1. Apple scab overwinters primarily in fallen leaves and in the soil.
            Disease development is favored by wet, cool weather that generally occurs in spring and early summer.

        2. Fungal spores are carried by wind, rain or splashing water from the ground to flowers, leaves or fruit.
            During damp or rainy periods, newly opening apple leaves are extremely susceptible to infection.
            The longer the leaves remain wet, the more severe the infection will be.
            Apple scab spreads rapidly between 55-75 degrees Fahrenheit.

        How to prevent/cure the disease:

        1. Choose resistant varieties when possible.

        2. Rake under trees and destroy infected leaves to reduce the number of fungal spores available to start the disease cycle over again next spring.

        3. Water in the evening or early morning hours (avoid overhead irrigation) to give the leaves time to dry out before infection can occur.

        4. Spread a 3- to 6-inch layer of compost under trees, keeping it away from the trunk, to cover soil and prevent splash dispersal of the fungal spores.""",

    'Grape___healthy': """ Crop: Grape  Disease: No disease

        Don't worry. Your crop is healthy. Keep it up !!!""",


    'Corn_(maize)___healthy': """ Crop: Corn(maize)  Disease: No disease

        Don't worry. Your crop is healthy. Keep it up !!!""",


    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': """ Crop : Grape   Disease: Leaf Spot""",


    'Orange___Haunglongbing_(Citrus_greening)': """ Crop: Orange  Disease: Citrus Greening

        Cause of disease:
        
        1. Huanglongbing (HLB) or citrus greening is the most severe citrus disease, currently devastating the citrus industry worldwide.
           The presumed causal bacterial agent Candidatus Liberibacter spp. affects tree health as well as fruit development, ripening and quality of citrus fruits and juice.

        How to prevent/cure the disease:

        1. In regions where disease incidence is low, the most common practices are avoiding the spread of infection by removal of symptomatic trees, protecting grove edges through intensive monitoring, use of pesticides, and biological control of the vector ACP.

        2. According to Singerman and Useche (2016), CHMAs coordinate insecticide application to control the ACP spreading across area-wide neighboring commercial citrus groves as part of a plan to address the HLB disease.

        3. In addition to foliar nutritional sprays, plant growth regulators were tested, unsuccessfully, to reduce HLB-associated fruit drop (Albrigo and Stover, 2015).""",


    'Peach___Bacterial_spot': """ <b>Crop</b>: Peach  Disease: Bacterial Spot

        Cause of disease:

        1. The disease is caused by four species of Xanthomonas (X. euvesicatoria, X. gardneri, X. perforans, and X. vesicatoria).
            In North Carolina, X. perforans is the predominant species associated with bacterial spot on tomato and X. euvesicatoria is the predominant species associated with the disease on pepper.

        2. All four bacteria are strictly aerobic, gram-negative rods with a long whip-like flagellum (tail) that allows them to move in water, which allows them to invade wet plant tissue and cause infection.

        How to prevent/cure the disease:

        1. The most effective management strategy is the use of pathogen-free certified seeds and disease-free transplants to prevent the introduction of the pathogen into greenhouses and field production areas.
            Inspect plants very carefully and reject infected transplants- including your own!

        2. In transplant production greenhouses, minimize overwatering and handling of seedlings when they are wet.

        3. Trays, benches, tools, and greenhouse structures should be washed and sanitized between seedlings crops.

        4. Do not spray, tie, harvest, or handle wet plants as that can spread the disease.""",


    'Pepper,_bell___Bacterial_spot': """ Crop: Pepper  Disease: Bacterial Spot

        Cause of disease:

        1. Bacterial spot is caused by several species of gram-negative bacteria in the genus Xanthomonas.

        2. In culture, these bacteria produce yellow, mucoid colonies.
            A "mass" of bacteria can be observed oozing from a lesion by making a cross-sectional cut through a leaf lesion, placing the tissue in a droplet of water, placing a cover-slip over the sample, and examining it with a microscope (~200X)..

        How to prevent/cure the disease:

        1. The primary management strategy of bacterial spot begins with use of certified pathogen-free seed and disease-free transplants.

        2. The bacteria do not survive well once host material has decayed, so crop rotation is recommended.
            Once the bacteria are introduced into a field or greenhouse, the disease is very difficult to control.

        3. Pepper plants are routinely sprayed with copper-containing bactericides to maintain a "protective" cover on the foliage and fruit.""",

    'Peach___healthy': """ Crop: Peach  Disease: No disease

        Don't worry. Your crop is healthy. Keep it up !!!""",

    'Pepper,_bell___healthy': """ Crop: Pepper  Disease: No disease

        Don't worry. Your crop is healthy. Keep it up !!!""",

    'Potato___healthy': """ Crop: Potato  Disease: No disease

        Don't worry. Your crop is healthy. Keep it up !!!""",

    'Raspberry___healthy': """ Crop: Raspberry  Disease: No disease

        Don't worry. Your crop is healthy. Keep it up !!!""",

    'Soybean___healthy': """ Crop: Soyabean  Disease: No disease

        Don't worry. Your crop is healthy. Keep it up !!!""",

    'Strawberry___healthy': """ Crop: Strawberry  Disease: No disease

        Don't worry. Your crop is healthy. Keep it up !!!""",

    'Tomato___healthy': """ Crop: Tomato  Disease: No disease

        Don't worry. Your crop is healthy. Keep it up !!!""",


    'Potato___Early_blight': """ Crop: Potato  Disease: Early Blight

        Cause of disease:

        1. Early blight (EB) is a disease of potato caused by the fungus Alternaria solani. It is found wherever potatoes are grown.

        2. The disease primarily affects leaves and stems, but under favorable weather conditions, and if left uncontrolled, can result in considerable defoliation and enhance the chance for tuber infection.
            Premature defoliation may lead to considerable reduction in yield.

        3. Primary infection is difficult to predict since EB is less dependent upon specific weather conditions than late blight.

        How to prevent/cure the disease:

        1. Plant only diseasefree, certified seed.

        2. Follow a complete and regular foliar fungicide spray program.

        3. Practice good killing techniques to lessen tuber infections.

        4. Allow tubers to mature before digging, dig when vines are dry, not wet, and avoid excessive wounding of potatoes during harvesting and handling.""",


    'Potato___Late_blight': """ Crop: Potato  Disease: Late Blight

        Late blight is a potentially devastating disease of potato, infecting leaves, stems and fruits of plants.
        The disease spreads quickly in fields and can result in total crop failure if untreated.
        Late blight of potato was responsible for the Irish potato famine of the late 1840s.

        Cause of disease:

        1. Late blight is caused by the oomycete Phytophthora infestans. Oomycetes are fungus-like organisms also called water molds, but they are not true fungi.

        2. There are many different strains of P. infestans. These are called clonal lineages and designated by a number code (i.e. US-23).
            Many clonal lineages affect both tomato and potato, but some lineages are specific to one host or the other.

        3. The host range is typically limited to potato and tomato, but hairy nightshade (Solanum physalifolium) is a closely related weed that can readily become infected and may contribute to disease spread. Under ideal conditions, such as a greenhouse, petunia also may become infected.

        How to prevent/cure the disease:

        1. Seed infection is unlikely on commercially prepared tomato seed or on saved seed that has been thoroughly dried.

        2. Inspect tomato transplants for late blight symptoms prior to purchase and/or planting, as tomato transplants shipped from southern regions may be infected.

        3. If infection is found in only a few plants within a field, infected plants should be removed, disced-under, killed with herbicide or flame-killed to avoid spreading through the entire field.""",


    'Squash___Powdery_mildew': """ Crop: Squash  Disease: Powdery mildew

        Cause of disease:

        1. Powdery mildew infections favor humid conditions with temperatures around 68-81° F.

        2. In warm, dry conditions, new spores form and easily spread the disease.

        3. Symptoms of powdery mildew first appear mid to late summer in Minnesota.
            The older leaves are more susceptible and powdery mildew will infect them first.

        4. Wind blows spores produced in leaf spots to infect other leaves.

        5. Under favorable conditions, powdery mildew can spread very rapidly, often covering all of the leaves.

        How to prevent/cure the disease:

        1. Apply fertilizer based on soil test results. Avoid over-applying nitrogen.

        2. Provide good air movement around plants through proper spacing, staking of plants and weed control.

        3. Once a week, examine five mature leaves for powdery mildew infection. In large plantings, repeat at 10 different locations in the field.

        4. If susceptible varieties are growing in an area where powdery mildew has resulted in yield loss in the past, fungicide may be necessary.""",


    'Strawberry___Leaf_scorch': """ Crop: Strawberry  Disease: Leaf Scorch

        Cause of disease:

        1. Scorched strawberry leaves are caused by a fungal infection which affects the foliage of strawberry plantings.
            The fungus responsible is called Diplocarpon earliana.

        2. Strawberries with leaf scorch may first show signs of issue with the development of small purplish blemishes that occur on the topside of leaves.

        How to prevent/cure the disease:

        1. Since this fungal pathogen over winters on the fallen leaves of infect plants, proper garden sanitation is key.

        2. This includes the removal of infected garden debris from the strawberry patch, as well as the frequent establishment of new strawberry transplants.

        3. The avoidance of waterlogged soil and frequent garden cleanup will help to reduce the likelihood of spread of this fungus.""",



    'Tomato___Bacterial_spot': """ <b>Crop</b>: Tomato  Disease: Bacterial Spot

        Cause of disease:

        1. The disease is caused by four species of Xanthomonas (X. euvesicatoria, X. gardneri, X. perforans, and X. vesicatoria).
            In North Carolina, X. perforans is the predominant species associated with bacterial spot on tomato and X. euvesicatoria is the predominant species associated with the disease on pepper.

        2. All four bacteria are strictly aerobic, gram-negative rods with a long whip-like flagellum (tail) that allows them to move in water, which allows them to invade wet plant tissue and cause infection.

        How to prevent/cure the disease:

        1. The most effective management strategy is the use of pathogen-free certified seeds and disease-free transplants to prevent the introduction of the pathogen into greenhouses and field production areas.
            Inspect plants very carefully and reject infected transplants- including your own!

        2. In transplant production greenhouses, minimize overwatering and handling of seedlings when they are wet.

        3. Trays, benches, tools, and greenhouse structures should be washed and sanitized between seedlings crops.

        4. Do not spray, tie, harvest, or handle wet plants as that can spread the disease""",


    'Tomato___Early_blight': """ Crop: Tomato  Disease: Early Blight

        Cause of disease:

        1. Early blight can be caused by two different closely related fungi, Alternaria tomatophila and Alternaria solani.

        2. Alternaria tomatophila is more virulent on tomato than A. solani, so in regions where A. tomatophila is found, it is the primary cause of early blight on tomato.
            However, if A.tomatophila is absent, A.solani will cause early blight on tomato.

        How to prevent/cure the disease:

        1. Use pathogen-free seed, or collect seed only from disease-free plants.

        2. Rotate out of tomatoes and related crops for at least two years.

        3. Control susceptible weeds such as black nightshade and hairy nightshade, and volunteer tomato plants throughout the rotation.

        4. Fertilize properly to maintain vigorous plant growth. Particularly, do not over-fertilize with potassium and maintain adequate levels of both nitrogen and phosphorus.

        5. Avoid working in plants when they are wet from rain, irrigation, or dew.

        6. Use drip irrigation instead of overhead irrigation to keep foliage dry.""",


    'Tomato___Late_blight': """ Crop: Tomato  Disease: Late Blight

        Late blight is a potentially devastating disease of tomato, infecting leaves, stems and fruits of plants.
        The disease spreads quickly in fields and can result in total crop failure if untreated.

        Cause of disease:

        1. Late blight is caused by the oomycete Phytophthora infestans. Oomycetes are fungus-like organisms also called water molds, but they are not true fungi.

        2. There are many different strains of P. infestans. These are called clonal lineages and designated by a number code (i.e. US-23).
            Many clonal lineages affect both tomato and potato, but some lineages are specific to one host or the other.

        3. The host range is typically limited to potato and tomato, but hairy nightshade (Solanum physalifolium) is a closely related weed that can readily become infected and may contribute to disease spread.
            Under ideal conditions, such as a greenhouse, petunia also may become infected.""",



    'Tomato___Leaf_Mold': """ Crop: Tomato  Disease: Leaf Mold

        Cause of disease:

        1. Leaf mold is caused by the fungus Passalora fulva (previously called Fulvia fulva or Cladosporium fulvum).
            It is not known to be pathogenic on any plant other than tomato.
            
        2. Leaf spots grow together and turn brown. Leaves wither and die but often remain attached to the plant.

        3. Fruit infections start as a smooth black irregular area on the stem end of the fruit.
            As the disease progresses, the infected area becomes sunken, dry and leathery.

        How to prevent/cure the disease:

        1. Use drip irrigation and avoid watering foliage.

        2. Space plants to provide good air movement between rows and individual plants.

        3. Stake, string or prune to increase airflow in and around the plant.

        4. Sterilize stakes, ties, trellises etc. with 10 percent household bleach or commercial sanitizer.

        5. Circulate air in greenhouses or tunnels with vents and fans and by rolling up high tunnel sides to reduce humidity around plants.

        6. Keep night temperatures in greenhouses higher than outside temperatures to avoid dew formation on the foliage.

        7. Remove crop residue at the end of the season. Burn it or bury it away from tomato production areas.""",


    'Tomato___Septoria_leaf_spot': """ Crop: Tomato  Disease: Leaf Spot

        Cause of disease:

        1. Septoria leaf spot is caused by a fungus, Septoria lycopersici. It is one of the most destructive diseases of tomato foliage and is particularly severe in areas where wet, humid weather persists for extended periods.

        How to prevent/cure the disease:

        1. Remove diseased leaves.

        2. Improve air circulation around the plants.

        3. Mulch around the base of the plants.

        4. Do not use overhead watering.

        5. Use fungicidal sprayes.""",



    'Tomato___Spider_mites Two-spotted_spider_mite': """ Crop: Tomato  Disease: Two-spotted spider mite

        Cause of disease:

        1. The two-spotted spider mite is the most common mite species that attacks vegetable and fruit crops.

        2. They have up to 20 generations per year and are favored by excess nitrogen and dry and dusty conditions.

        3. Outbreaks are often caused by the use of broad-spectrum insecticides which interfere with the numerous natural enemies that help to manage mite populations.\

        How to prevent/cure the disease:

        1. Avoid early season, broad-spectrum insecticide applications for other pests.

        2. Do not over-fertilize.

        3. Overhead irrigation or prolonged periods of rain can help reduce populations.""",


    'Tomato___Target_Spo': """ Crop: Tomato  Disease: Target Spot

        Cause of disease:

        1. The fungus causes plants to lose their leaves; it is a major disease. If infection occurs before the fruit has developed, yields are low.

        2. This is a common disease on tomato in Pacific island countries. The disease occurs in the screen house and in the field.

        How to prevent/cure the disease:

        1. Remove a few branches from the lower part of the plants to allow better airflow at the base.

        2. Remove and burn the lower leaves as soon as the disease is seen, especially after the lower fruit trusses have been picked.

        3. Keep plots free from weeds, as some may be hosts of the fungus.

        4. Do not use overhead irrigation; otherwise, it will create conditions for spore production and infection.""",


    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': """ Crop: Tomato  Disease: Yellow Leaf Curl Virus

        Cause of disease:

        1. TYLCV is transmitted by the insect vector Bemisia tabaci in a persistent-circulative nonpropagative manner.
            The virus can be efficiently transmitted during the adult stages.
        2. This virus transmission has a short acquisition access period of 15–20 minutes, and latent period of 8–24 hours.

        How to prevent/cure the disease:

        1. Currently, the most effective treatments used to control the spread of TYLCV are insecticides and resistant crop varieties.

        2. The effectiveness of insecticides is not optimal in tropical areas due to whitefly resistance against the insecticides; therefore, insecticides should be alternated or mixed to provide the most effective treatment against virus transmission.

        3. Other methods to control the spread of TYLCV include planting resistant/tolerant lines, crop rotation, and breeding for resistance of TYLCV. As with many other plant viruses, one of the most promising methods to control TYLCV is the production of transgenic tomato plants resistant to TYLCV.""",


    'Tomato___Tomato_mosaic_virus': """ Crop: Tomato  Disease: Mosaic Virus

        Cause of disease:

        1. Tomato mosaic virus and tobacco mosaic virus can exist for two years in dry soil or leaf debris, but will only persist one month if soil is moist.
            The viruses can also survive in infected root debris in the soil for up to two years.

        2. Seed can be infected and pass the virus to the plant but the disease is usually introduced and spread primarily through human activity.
            The virus can easily spread between plants on workers' hands, tools, and clothes with normal activities such as plant tying, removing of suckers, and harvest.

        3. The virus can even survive the tobacco curing process, and can spread from cigarettes and other tobacco products to plant material handled by workers after a cigarette

        How to prevent/cure the disease:

        1. Purchase transplants only from reputable sources. Ask about the sanitation procedures they use to prevent disease.

        2. Inspect transplants prior to purchase. Choose only transplants showing no clear symptoms.

        3. Avoid planting in fields where tomato root debris is present, as the virus can survive long-term in roots.

        4. Wash hands with soap and water before and during the handling of plants to reduce potential spread between plants."""
}


st.title("Smart Cultivation and Prediction System for Agriculture")
menu = ["Login","SignUp"]
choice = st.selectbox("Menu",menu)

if choice == "Login":
        username= st.text_input("User Name")
        password= st.text_input("Password", type='password')
        if st.button("Login"):
                if not (username):
                        st.warning("Enter Valid Username")
                elif not (password):
                        st.warning("Enter Valid Password")
                else:
                        st.success("Logged In as ",format(username))
                        task = st.selectbox("Navigate",["Crop Suggestion","Disease Identification",])
                        if task == "Disease Identification":
                                st.subheader("Disease Identification")
                                def pred(image1):
                                        model = load_model('models/AlexNetModel.hdf5')


                                        output_dict = {'Apple___Apple_scab': 0,
                                                       'Apple___Black_rot': 1,
                                                       'Apple___Cedar_apple_rust': 2,
                                                       'Apple___healthy': 3,
                                                       'Blueberry___healthy': 4,
                                                       'Cherry_(including_sour)___Powdery_mildew': 5,
                                                       'Cherry_(including_sour)___healthy': 6,
                                                       'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7,
                                                       'Corn_(maize)___Common_rust_': 8,
                                                       'Corn_(maize)___Northern_Leaf_Blight': 9,
                                                       'Corn_(maize)___healthy': 10,
                                                       'Grape___Black_rot': 11,
                                                       'Grape___Esca_(Black_Measles)': 12,
                                                       'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13,
                                                       'Grape___healthy': 14,
                                                       'Orange___Haunglongbing_(Citrus_greening)': 15,
                                                       'Peach___Bacterial_spot': 16,
                                                       'Peach___healthy': 17,
                                                       'Pepper,_bell___Bacterial_spot': 18,
                                                       'Pepper,_bell___healthy': 19,
                                                       'Potato___Early_blight': 20,
                                                       'Potato___Late_blight': 21,
                                                       'Potato___healthy': 22,
                                                       'Raspberry___healthy': 23,
                                                       'Soybean___healthy': 24,
                                                       'Squash___Powdery_mildew': 25,
                                                       'Strawberry___Leaf_scorch': 26,
                                                       'Strawberry___healthy': 27,
                                                       'Tomato___Bacterial_spot': 28,
                                                       'Tomato___Early_blight': 29,
                                                       'Tomato___Late_blight': 30,
                                                       'Tomato___Leaf_Mold': 31,
                                                       'Tomato___Septoria_leaf_spot': 32,
                                                       'Tomato___Spider_mites Two-spotted_spider_mite': 33,
                                                       'Tomato___Target_Spot': 34,
                                                       'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35,
                                                       'Tomato___Tomato_mosaic_virus': 36,
                                                       'Tomato___healthy': 37}

                                        output_list = list(output_dict.keys())


                                        default_image_size = tuple((256, 256))
                                        def convert_image_to_array(image_dir):
                                            try:
                                                image = cv2.imread(image_dir)
                                                if image is not None :
                                                    image = cv2.resize(image, default_image_size)   
                                                    return img_to_array(image)
                                                else :
                                                    return np.array([])
                                            except Exception as e:
                                                print(f"Error : {e}")
                                                return None


                                        img = image1
                                        img = image.img_to_array(img)
                                        img = np.expand_dims(img, axis=0)
                                        img = img/255
                                        prediction = model.predict(img)

                                        prediction_flatten = prediction.flatten()
                                        
                                        max_val_index = np.argmax(prediction_flatten)
                                        result = output_list[max_val_index]

                                            
                                        accuracy =prediction_flatten[max_val_index]*100
                                        accuracy = float("{:.3f}".format(accuracy))
                                            

                                        return result, accuracy;

                                st.title("Plant Disease prediction")

                                uploaded_file = st.file_uploader("Choose an image...", type="jpg")
                                if uploaded_file is not None:
                                    imageee = Image.open(uploaded_file)
                                    imagee = imageee.resize((224, 224))
                                    r,a = pred(imagee)
                                    st.image(imageee, caption='Uploaded Image.', use_column_width=True)
                                    if(st.button("Predict")):
                                        st.write('%s (%.2f%%)' % (r,a))
                                        if a > 80:
                                                predictin = Markup(str(disease_dic[r]))
                                                st.write(predictin)
                                        else:
                                                st.warning("Warning! prediction may not be accurate")
                                                #url = 'http://localhost:8501'
                                                #webbrowser.open_new_tab(url)

                                                
                        elif task == "Crop Suggestion":
                                db1 = firestore.Client.from_service_account_json("serviceAccountKey.json")
                                docs1 = list(db1.collection(u'username').stream())
                                docs_dict1 = list(map(lambda x: x.to_dict(), docs1))
                                df1 = pd.DataFrame(docs_dict1)
                                #st.write(df1)
                                n = df1['n']
                                n = n.iloc[0]
                                p = df1['p']
                                p =p.iloc[0]
                                k = df1['k']
                                k =k.iloc[0]
                                soil_moisture = df1['soil moisture']
                                soil_moisture =soil_moisture.iloc[0]
                                soil_temperature = df1['soil temperature']
                                soil_temperature =soil_temperature.iloc[0]
                                surrounding_temperature = df1['surrounding temperature']
                                surrounding_temperature =surrounding_temperature.iloc[0]
                                surrounding_humidity = df1['surrounding humidity']
                                surrounding_humidity =surrounding_humidity.iloc[0]
                                rainfall = df1['rainfall']
                                rainfall = rainfall.iloc[0]
                                ph = df1['pH']
                                ph = ph.iloc[0]
                                city = df1['city']
                                city = city.iloc[0]
                                asach = pd.DataFrame(np.array([n,p,k,soil_moisture,soil_temperature,surrounding_temperature,surrounding_humidity]))
                                #st.write(asach) 
                                st.subheader("Crop Suggestion")
                                crop_recommendation_model_path = 'models/RandomForest.pkl'
                                crop_recommendation_model = pickle.load(
                                open(crop_recommendation_model_path, 'rb'))


                                N = st.number_input('Nitrogen',min_value=0, max_value=700, value=n, step=1)
                                P = st.number_input('Phosphorus',min_value=0, max_value=500, value=p, step=1)
                                K = st.number_input('Potassium',min_value=0, max_value=500, value=k, step=1)
                                ph = st.number_input('pH',min_value=0, max_value=14, value=ph, step=1)
                                rainfall = st.number_input('Rainfall(mm)',min_value=0, max_value=1000, value=rainfall, step=1)
                                soil_m = st.number_input('Soil Moisture',min_value=0, max_value=1000, value=soil_moisture, step=1)
                                soil_t = st.number_input('Soil Temperature',min_value=0, max_value=1000, value=soil_temperature, step=1)
                                #temperature = 
                                #humidity = 
                                city_name = st.text_input("City", value=city)

                                def preds(N, P, K, ph,city_name, rainfall):

                                        api_key = '9d7cde1f6d07ec55650544be1631307e'
                                        base_url = "http://api.openweathermap.org/data/2.5/weather?"
                                        complete_url = base_url + "appid=" + api_key + "&q=" + city_name
                                        response = requests.get(complete_url)
                                        x = response.json()
                                        y = x['main']
                                        temperature = round((y["temp"] - 273.15), 2)
                                        humidity = y["humidity"]
                                        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
                                        my_prediction = crop_recommendation_model.predict(data)
                                        final_prediction = my_prediction[0]

                                        return final_prediction,temperature,humidity






                                #temperature = round((x["temp"] - 273.15), 2)
                                #humidity = x["humidity"]
                                session_state = SessionState.get(name = "", button_start = False)
                                button_start = st.button('Predict')
                                if button_start:
                                        session_state.button_start = True
                                        st.success("Success")
                                        final_prediction,temperature,humidity = preds(N, P, K, ph,city_name, rainfall)
                                        st.write("The city of " + city_name)
                                        st.write('Temperature : {}C'.format(temperature))
                                        st.write('humidity : {}'.format(humidity)) 
                                        st.subheader("The Suggested crop is :")
                                        st.write(final_prediction)
                                        crop_name = final_prediction
                                        df = pd.read_csv('Data/fertilizer.csv')
                                        nr = df[df['Crop'] == crop_name]['N'].iloc[0]
                                        pr = df[df['Crop'] == crop_name]['P'].iloc[0]
                                        kr = df[df['Crop'] == crop_name]['K'].iloc[0]
                                        n = nr - N
                                        p = pr - P
                                        k = kr - K
                                        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
                                        max_value = temp[max(temp.keys())]
                                        if max_value == "N":
                                                if n < 0:
                                                        key = 'NHigh'
                                                else:
                                                        key = "Nlow"
                                        elif max_value == "P":
                                                if p < 0:
                                                        key = 'PHigh'
                                                else:
                                                        key = "Plow"
                                        else:
                                                if k < 0:
                                                        key = 'KHigh'
                                                else:
                                                        key = "Klow"

                                        response = Markup(str(fertilizer_dic[key]))
                                        st.write(response)
                                                
elif choice == "SignUp":
        new_user = st.text_input("Username")
        new_password = st.text_input("Password",type='password')
        confirmpassword = st.text_input("Confirm Password",type='password')

        if st.button("Signup"):
            if not (new_user):
                st.warning("Enter Valid Username")
            elif not (new_password):
                st.warning("Enter Valid Password")
            elif (new_password!= confirmpassword):
                st.warning("Password does not match")
            else:
                db = firestore.Client.from_service_account_json("serviceAccountKey.json")
                data={"username":new_user,"password":new_password}
                doc_name=new_user+"_final_record"
                db.collection(new_user).document(doc_name).set(data)
                st.success("You have successfully created a valid Account")
                st.info("Go to Login Menu to login")
            
