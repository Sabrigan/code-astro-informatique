#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
le programme suivant est le main, il sera le programme principal permet
tant d'appeler tout les autres. Il contient plusieurs sous programme:

P1: - sert a télécharger depuis l'url:  http://cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv/
    - enregistre les données dans un dataframe
    - selectionne les données interessantes pour un algo de kmeans ou 
    de knn. 
    - supprime le fichier telechargé quand les données sont copiées
    - enregistre le dataframe sur un fichier unique. 
comporte une boucle sur tout les éléments contenus dans l'url. On verra 
si un seul dataframe suffit.
Possibilité de supplément: 
		- attribuer des catégorie classique a chaque etoile en fonction 
		du diagramme H-R);
		- fusionner avec la base des données d'exosystèmes pour visuali-
		ser où ils sont sur le diagramme H-R.   

P2: division (split) des données en training, test, le reste. on prendra
10% du total et on le splitera entre training et test. 
 
P3: Entrainement du modele de knn ou de kmeans suivant. Les resultatq 
peuvent se traduire par des couleurs sur un graphique (avec knn on peut 
attribuer k couleurs, avec kmeans on verra).

P4: affichage des resultats sous forme d'un diagramme HR. Pour la compa-
raison on peut faire un schéma avec les catégories traditionnelles et un
avec les catégroies trouvées. 
"""
def clear_df(df):
	clear_list=['random_index','ref_epoch','ra_error','dec_error',
	'parallax_error','parallax_over_error','pmra_error','pmdec_error',
	'ra_dec_corr','ra_parallax_corr','ra_pmra_corr','ra_pmdec_corr',
	'dec_parallax_corr','dec_pmra_corr','dec_pmdec_corr','parallax_pmra_corr',
	'parallax_pmdec_corr','pmra_pmdec_corr','astrometric_n_obs_al',
	'astrometric_n_obs_ac','astrometric_n_good_obs_al','astrometric_n_bad_obs_al',
	'astrometric_gof_al','astrometric_chi2_al','astrometric_excess_noise',
	'astrometric_params_solved','astrometric_primary_flag','astrometric_weight_al',
	'astrometric_pseudo_colour_error','mean_varpi_factor_al','astrometric_matched_observations',
	'visibility_periods_used','astrometric_sigma5d_max','frame_rotator_object_type',
	'matched_observations','phot_g_n_obs','phot_g_mean_flux_error',
	'phot_g_mean_flux_over_error','phot_bp_n_obs','phot_bp_mean_flux_error',
	'phot_bp_mean_flux_over_error','phot_bp_mean_mag','phot_rp_n_obs',
	'phot_rp_mean_flux_error','phot_rp_mean_flux_over_error','phot_rp_mean_mag',
	'phot_bp_rp_excess_factor','phot_proc_mode','radial_velocity_error',
	'rv_nb_transits','rv_template_logg','phot_variable_flag','priam_flags',
	'teff_percentile_lower','teff_percentile_upper','a_g_percentile_lower',
	'a_g_percentile_upper','e_bp_min_rp_percentile_lower','e_bp_min_rp_percentile_upper',
	'flame_flags','radius_percentile_lower','radius_percentile_upper',
	'lum_percentile_lower','lum_percentile_upper']
	df=df.drop(clear_list, axis=1)
	return df
	
def clear_df_exopl(df):
	clear_list_exopl=['mass_error_min', 'mass_error_max','mass_sini_error_min', 
	'mass_sini_error_max','radius_error_min', 'radius_error_max',
	'orbital_period_error_min', 'orbital_period_error_max','semi_major_axis_error_min',
	'semi_major_axis_error_max','eccentricity_error_min','eccentricity_error_max',
	'inclination_error_min','inclination_error_max', 'omega_error_min', 
	'omega_error_max','tperi_error_min', 'tperi_error_max', 'tconj_error_min',
	'tconj_error_max','tzero_tr_error_min','tzero_tr_error_max',
	'tzero_tr_sec_error_min','tzero_tr_sec_error_max','lambda_angle_error_min',
	'lambda_angle_error_max','impact_parameter_error_min',
	'impact_parameter_error_max','tzero_vr_error_min', 'tzero_vr_error_max', 
	'k_error_min','k_error_max','temp_calculated_error_min','temp_calculated_error_max',
	'geometric_albedo_error_min','geometric_albedo_error_max','star_distance_error_min',
	'star_distance_error_max','star_metallicity_error_min', 'star_metallicity_error_max',
	'star_mass_error_min', 'star_mass_error_max','star_radius_error_min', 
	'star_radius_error_max','star_age_error_min', 'star_age_error_max',
	'star_teff_error_min', 'star_teff_error_max']	
	df=df.drop(clear_list_exopl, axis=1)	
	return df

def scrapp(url):
	from bs4 import BeautifulSoup as bs
	import urllib.request
	import gc

	page = urllib.request.urlopen(url,timeout=20)
	soup = bs(page)
	liste=[]
	for a in soup.find_all('a', href=True):
		if "Gaia" in a['href']:
			liste.append(a['href'])
	del(page,soup)
	gc.collect(0)
	gc.collect(1)
	gc.collect(2)
	return liste

def upload_file(liste,url1,i):
	import requests as req
	import gc
	url = url1+liste[i]
	file1 = req.get(url, allow_redirects=True)
	f=open(liste[i], 'wb')
	f.write(file1.content)
	f.close()
	del(f,file1,url,liste)
	gc.collect(0)
	gc.collect(1)
	gc.collect(2)

def P1():
	import gzip
	import shutil
	import os
	import pandas as pd
	from progress.bar import Bar
	import gc
	import numpy as np
	
	# a faire: le scrapping de la page pour avoir la liste de tt les fichiers
	url = 'http://cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv/'
	liste_fichier=scrapp(url)
	
	max_donnee=50
	
	liste_rand=np.random.choice(liste_fichier, max_donnee)
	
	bar = Bar('importation',max=max_donnee)
	# on prend le premier element
	# on telecharge un fichier en archive et on l'enregistre
	i=0
	upload_file(liste_rand,url,i)
		
	#creer un df de pandas depuis le fichier 
	df = pd.read_csv(liste_rand[i],compression='gzip',header=0)
	
	#netoyage de df
	clear_df(df)
	
	bar.next()
	#on va supprimer le ficher gzip
	os.remove(liste_rand[i])
	
	#----------------------------------------------------
	while i<len(liste_rand)-1:
		#i<len(liste_fichier)-1:
		# on telecharge un autre fichier en archive et on l'enregistre
		i=i+1

		upload_file(liste_rand,url,i)

		#if i>1 and i%100==1:
		#	df = pd.read_csv('myFile.csv', header=0)
		
		#creer un df de pandas depuis le fichier csv
		df2 = pd.read_csv(liste_rand[i],compression = 'gzip', header=0)
		clear_df(df2)
		
		#on va supprimer le ficher archive
		os.remove(liste_rand[i])

		#concatener deux df 
		df=pd.concat([df,df2], ignore_index= True)
		del(df2)
		gc.collect(0)
		gc.collect(1)
		gc.collect(2)
		#if i%100==0:
		#	df.to_csv('myFile.csv')
		#	del(df)
		#	gc.collect(0)
		#	gc.collect(1)
		#	gc.collect(2)
		bar.next()
	bar.finish()
	del(bar)
	gc.collect()
	#------------------------------------------------------------
	print('traitement des données etoiles')


	#ecrire le nouveau df dans un fichier
	df.to_csv('myFile.csv')
	#del(df)
	#gc.collect()

	
	print('traitement des données planetes')
	#créer un nouveau df du catalogue des exoplanetes
	file_exopl='exoplanet.eu_catalog.csv'
	df_exopl=pd.read_csv(file_exopl, header=0)
	
	#netoyage de df_exoplanete
	clear_df_exopl(df_exopl)
	
	#traitement des données avec le confirmed = 1, candidate = 0.2 , controversial = 0.5 , retracted = 0
	df_exopl.loc[df_exopl.planet_status=='Confirmed', 'planet_status']=1
	df_exopl.loc[df_exopl.planet_status=='Controversial', 'planet_status']=0.5
	df_exopl.loc[df_exopl.planet_status=='Candidate', 'planet_status']=0.2
	df_exopl.loc[df_exopl.planet_status=='Retracted', 'planet_status']=0
	
	#print(df_exopl.columns)
	df_exopl.to_csv('myFile2.csv')

	#fusion des deux type de données avec  une colonne pour l'observation 
	# ou pas(0 non, 1 oui), une colonne pour planet_status (si plusieurs 
	# on multiplie), une colonne avec le nombre de planetes.
	
	df.insert(len(df.columns),"obs4exopl_status",0)
	df.insert(len(df.columns),"exopl_status",0)
	df.insert(len(df.columns),"exopl_nb",0)

	lst_star_ra=df.ra.to_numpy()
	bar1 = Bar('existance dans etoile',max=len(df_exopl))
	for i in range(len(df_exopl)):
		if df_exopl.ra[i] in lst_star_ra:
			print('correspondance',)
			lst_crpd_ra = np.where(lst_star_ra==df_exopl.ra[i])[0].tolist()
			bar2 = Bar('correspondance avec 1 etoile',max=len(lst_crpd_ra))
			for j in lst_crpd_ra:
				if df_exopl.dec[i]==df.dec[j]:
					df.at[j, 'obs4exopl_status']=1
					if df.exopl_status[j]==0:
						df.at[j, 'exopl_status']= df_exopl.planet_status[i]
					else:
						df.at[j,'exopl_status']= df_exopl.planet_status[i]*df.exopl_status[j]
					df.at[j,'exopl_nb']=df.exopl_nb[j]+1
				bar2.next()
			bar2.finish()
			del(lst_crpd_ra)
			gc.collect(0)
			gc.collect(1)
			gc.collect(2)
		bar1.next()
	bar1.finish()

	df.to_csv('myFile3.csv')
	#del(df,bar1,bar2)
	#gc.collect(0)
	#gc.collect(1)
	#gc.collect(2)

def P4():
	import matplotlib.pyplot as plt
	import seaborn as sns	
	import matplotlib.image as image
	import pandas as pd
	
	# echelle logarithmique pour la luminosité
	# faire un graphe avec luminosté (lum_val) en ordonnée et couleur ou 
	# temperature en abscisse (astrometric_pseudo_colour ou teff_val)
	# borne ordonnée: 0,000001 - 1 000 000 (echelle log)
	# borne abscisse (couleur): -0,35 - 2,25 (0,1 par graduation)
	# borne abscisse (teff): 3000K-300000K (graduation log)
	
	#réimportation du fichier de donnée
	print('preparation de la sortie')
	file_star='myFile3.csv'
	df_star=pd.read_csv(file_star, header=0)
	
	# fichier en filigrane
	file1='diagramme_HR_blanc.png'
	im = image.imread(file1)
	
	#taille en dure pour l'image en filigrane
	sns.set(rc = {'figure.figsize':(7.28,9)})
	
	fig, ax1 = plt.subplots()

	#palette divergente pour les temperature des etoiles
	pal1= sns.diverging_palette(20, 225, as_cmap=True)
	g=sns.scatterplot(ax =ax1, x='astrometric_pseudo_colour', y='lum_val', hue='teff_val', palette=pal1, data=df_star, sizes=(5))
	
	# titre de la figure
	g.set_title('luminosité=f(couleur)')
	
	#limite pour x et y
	plt.xlim((-0.35,2.25))
	plt.ylim((0.000001,1000000))
	
	#luminosité en echelle logaritmique
	g.set_yscale("log")
	
	
	
	fig.figimage(im,10,10, zorder=3, alpha=0.1)
	plt.show()
"""
def P3():
	
def P2():
"""
def P6():
	import pandas as pd
	from bokeh.plotting import figure, show
	from bokeh.models import CustomJS, Div, Row
	from bokeh.palettes import RdYlBu10
	from bokeh.transform import log_cmap
	from bokeh.io import curdoc
	import matplotlib.pyplot as plt

	
	# echelle logarithmique pour la luminosité
	# faire un graphe avec luminosté (lum_val) en ordonnée et couleur ou 
	# temperature en abscisse (astrometric_pseudo_colour ou teff_val)
	# borne ordonnée: 0,000001 - 1 000 000 (echelle log)
	# borne abscisse (couleur): -0,35 - 2,25 (0,1 par graduation)
	# borne abscisse (teff): 3000K-300000K (graduation log)
	
	#réimportation du fichier de donnée
	print('preparation de la sortie')
	file_star='myFile3.csv'
	df_star=pd.read_csv(file_star, header=0)
	
		
	#mise en place du plot
	g=figure(title='luminosité=f(couleur)', 
			x_axis_label='couleur', y_axis_label='luminosité',
			x_axis_type='linear', y_axis_type='log',
			x_range=(-0.35, 2.25), y_range=(0.000001,1000000),
			plot_width=655, plot_height=816)
	
	
	#palette
	invert_palette=RdYlBu10[::-1]
	mapper = log_cmap(field_name='teff_val', palette=invert_palette[:] ,low=2000 ,high=40000)
	
	# fichier en filigrane
	#file1='diagramme_HR_blanc.png'
	#d1 = Div(text = '<div style="position: absolute; left:-678px; top:-12px"><img src=' + file1 + ' style="width:500px; height:600px; opacity: 0.5"></div>')
	#g.image_url(url=['/static/diagramme_HR_blanc.png'], x=[0], y=[0], w=[720], h=[820], anchor="bottom_left")
	
	#curdoc().add_root(g)
	
	g.dot(x='astrometric_pseudo_colour', y='lum_val', source=df_star, color=mapper)
	
	#show(Row(g,d1))
	#show(Row(d1,g))
	show(g)
	
def P5():
	import pandas as pd
	import matplotlib.pyplot as plt
	from bokeh.palettes import RdYlBu10
	from bokeh.transform import log_cmap
	
	# echelle logarithmique pour la luminosité
	# faire un graphe avec luminosté (lum_val) en ordonnée et couleur ou 
	# temperature en abscisse (astrometric_pseudo_colour ou teff_val)
	# borne ordonnée: 0,000001 - 1 000 000 (echelle log)
	# borne abscisse (couleur): -0,35 - 2,25 (0,1 par graduation)
	# borne abscisse (teff): 3000K-300000K (graduation log)
	
	#réimportation du fichier de donnée
	print('preparation de la sortie')
	file_star='myFile3.csv'
	df_star=pd.read_csv(file_star, header=0)
	
	#palette
	invert_palette=RdYlBu10[::-1]
	mapper = log_cmap(field_name='teff_val', palette=invert_palette[:] ,low=2000 ,high=40000)
	
	# fichier en f
	img = plt.imread("diagramme_HR_blanc.png")
	fig, ax = plt.subplots()
	ax.imshow(img)
	ax.scatter(x=df_star.astrometric_pseudo_colour, y=df_star.lum_val)
	plt.show()	
	"""
	#mise en place du plot
	g=figure(title='luminosité=f(couleur)', 
			x_axis_label='couleur', y_axis_label='luminosité',
			x_axis_type='linear', y_axis_type='log',
			x_range=(-0.35, 2.25), y_range=(0.000001,1000000),
			plot_width=655, plot_height=816)
	
	
	#palette
	invert_palette=RdYlBu10[::-1]
	mapper = log_cmap(field_name='teff_val', palette=invert_palette[:] ,low=2000 ,high=40000)
	
	# fichier en filigrane
	file1='diagramme_HR_blanc.png'
	d1 = Div(text = '<div style="position: absolute; left:-678px; top:-12px"><img src=' + file1 + ' style="width:500px; height:600px; opacity: 0.5"></div>')
	#g.image_url(url=['/static/diagramme_HR_blanc.png'], x=[0], y=[0], w=[720], h=[820], anchor="bottom_left")
	
	#curdoc().add_root(g)
	
	g.dot(x='astrometric_pseudo_colour', y='lum_val', source=df_star, color=mapper)
	
	show(Row(g,d1))
	#show(Row(d1,g))
	#show(g)
"""
	
def main():
	print('importation et traitement des données ')
	P1()
	print('spliting entre données d entrainement, de test et de d etude')
	"""
	P2()
	print('entrainement du modele + resultat')
	P3()
	"""
	#print('print des resultats seaborn')
	#P4()
	print('print des resultats bokeh')
	P6()
	#print('print des resultats matplot')
	#P5()
	
	
if __name__ == "__main__":
	main()
