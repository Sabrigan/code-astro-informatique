#!/usr/bin/python
from bs4 import BeautifulSoup as bs
import numpy as np
import pandas as pd
import urllib.request
import requests as req
import os
import gzip
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

url = 'http://cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv/'

page = urllib.request.urlopen(url,timeout=20)

soup = bs(page)

liste_fichier=[]

for a in soup.find_all('a', href=True):
	if "Gaia" in a['href']:
		liste_fichier.append(a['href'])
max_d=5

liste_hazar=np.random.choice(liste_fichier, max_d)

print(liste_hazar)

"""
#print(len(liste_fichier))

i=0
#print(liste_fichier[i])
while i <len(liste_fichier)-1:
	i=i+1
	#print(liste_fichier[i])

url=url+liste_fichier[i]
file1 = req.get(url, allow_redirects=True)
open(liste_fichier[i], 'wb').write(file1.content)

df = pd.read_csv(liste_fichier[i], compression='gzip', header = 0)


#os.remove(liste_fichier[i])

#for i in range(1):
i=1
url = 'http://cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv/'
#print(url,i)
url = url+liste_fichier[i]

file2 = req.get(url, allow_redirects=True)
open(liste_fichier[i], 'wb').write(file2.content)

df2 = pd.read_csv(liste_fichier[i], compression='gzip', header = 0)
	
os.remove(liste_fichier[i])
	
df=pd.concat([df,df2], ignore_index= True)
"""
"""
file_star='myFile.csv'
df_star=pd.read_csv(file_star, header=0)
#print('df_star est de type', type(df_star))
"""
"""
file_exopl='exoplanet.eu_catalog.csv'

df_exopl=pd.read_csv(file_exopl, header=0)
df_exopl=df_exopl.drop(['mass_error_min', 'mass_error_max','mass_sini_error_min', 'mass_sini_error_max'], axis=1)
df_exopl=df_exopl.drop(['radius_error_min', 'radius_error_max','orbital_period_error_min', 'orbital_period_error_max'], axis=1)
df_exopl=df_exopl.drop(['semi_major_axis_error_min','semi_major_axis_error_max','eccentricity_error_min','eccentricity_error_max'], axis=1)
df_exopl=df_exopl.drop(['inclination_error_min','inclination_error_max', 'omega_error_min', 'omega_error_max'], axis=1)
df_exopl=df_exopl.drop(['tperi_error_min', 'tperi_error_max', 'tconj_error_min','tconj_error_max'], axis=1)
df_exopl=df_exopl.drop(['tzero_tr_error_min','tzero_tr_error_max','tzero_tr_sec_error_min','tzero_tr_sec_error_max'], axis=1)
df_exopl=df_exopl.drop(['lambda_angle_error_min','lambda_angle_error_max','impact_parameter_error_min','impact_parameter_error_max'], axis=1)
df_exopl=df_exopl.drop(['tzero_vr_error_min', 'tzero_vr_error_max', 'k_error_min','k_error_max'], axis=1)
df_exopl=df_exopl.drop(['temp_calculated_error_min','temp_calculated_error_max','geometric_albedo_error_min','geometric_albedo_error_max'], axis=1)
df_exopl=df_exopl.drop(['star_distance_error_min','star_distance_error_max','star_metallicity_error_min', 'star_metallicity_error_max'], axis=1)
df_exopl=df_exopl.drop(['star_mass_error_min', 'star_mass_error_max','star_radius_error_min', 'star_radius_error_max'], axis=1)
df_exopl=df_exopl.drop(['star_age_error_min', 'star_age_error_max','star_teff_error_min', 'star_teff_error_max'], axis=1)




#traitement des données avec le confirmed = 1, candidate = 0.2 , controversial = 0.5 , retracted = 0
df_exopl.loc[df_exopl.planet_status=='Confirmed', 'planet_status']=1
df_exopl.loc[df_exopl.planet_status=='Controversial', 'planet_status']=0.5
df_exopl.loc[df_exopl.planet_status=='Candidate', 'planet_status']=0.2
df_exopl.loc[df_exopl.planet_status=='Retracted', 'planet_status']=0

#print(df_exopl.columns)
print(df_exopl.columns)
df_exopl.to_csv('myFile2.csv')
"""
"""
file_planet='myFile2.csv'
df_planet=pd.read_csv(file_planet, header=0)
#print(df_star.columns)
#print(df_planet.columns)
"""
"""
# fusion avec le catalogue exoplanet et df
on va prendre comme point de comparaison ra, dec et teff pour 3eme 
verif. 
on va mettre d'abord les ra et dec au même niveau de précision.
on va creer une colonne pour l'observation ou pas(0 non, 1 oui), une colonne pour 
planet_status (si plusieurs on multiplie), une colonne avec le nombre 
de planetes.
"""
"""
df_star.insert(len(df_star.columns),"obs4exopl_status",0)
df_star.insert(len(df_star.columns),"exopl_status",0)
df_star.insert(len(df_star.columns),"exopl_nb",0)

lst_star_ra=df_star.ra.to_numpy()


for i in range(len(df_planet)):
	if df_planet.ra[i] in lst_star_ra:
		print('correspondance',)
		lst_crpd_ra = np.where(lst_star_ra==df_planet.ra[i])[0].tolist()
		for j in lst_crpd_ra:
			if df_planet.dec[i]==df_star.dec[j]:
				df_star.at[j, 'obs4exopl_status']=1
				if df_star.exopl_status[j]==0:
					df_star.at[j, 'exopl_status']= df_planet.planet_status[i]
				else:
					df_star.at[j,'exopl_status']= df_planet.planet_status[i]*df_star.exopl_status[j]
				df_star.at[j,'exopl_nb']=df_star.exopl_nb[j]+1

df_star.to_csv('myFile3.csv')


file_star='myFile3.csv'
df_star=pd.read_csv(file_star, header=0)

#echelle logarithmique pour la luminosité
	# faire un graphe avec luminosté (lum_val) en ordonnée et couleur ou 
	# temperature en abscisse (astrometric_pseudo_colour ou teff_val)
	# borne ordonnée: 0.000001 - 1 000 000 (echelle log)
	# borne abscisse (couleur): -0,35 - 2,25 (0,1 par graduation)
	# borne abscisse (teff): 3000K-300000K (graduation log)
import matplotlib.image as image


file1='diagramme_HR_blanc.png'
im = image.imread(file1)

sns.set(rc = {'figure.figsize':(7.28,9)})

fig, ax1 = plt.subplots()

pal1= sns.diverging_palette(20, 225, as_cmap=True)

g=sns.scatterplot(ax =ax1, x='astrometric_pseudo_colour', y='lum_val', hue='teff_val', palette=pal1, data=df_star, style='exopl_status')
g.set_yscale("log")
g.set_title('luminosité=f(couleur)')

plt.xlim((-0.35,2.25))
plt.ylim((0.000001,1000000))

fig.figimage(im,10,10, zorder=3, alpha=0.1)

plt.show()
"""



