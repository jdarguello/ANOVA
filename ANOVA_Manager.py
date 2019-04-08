#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from outliers import smirnov_grubbs as grubbs
import pandas as pd
import numpy as np
from scipy.stats import invgauss
import math
import matplotlib.pyplot as plt
import openpyxl

data = np.array([2.9, 2.2, 2.2, 2.1, 2.3, 2.2, 2.7, 2.3])

g = grubbs.test(data, alpha=0.05)

class TxtData():
	"""
		Obtiene datos generales
	"""
	
	def txtinput(self):
		with open("Generalidades.txt", "r") as file:
			frases = file.read().split("\n")
			data = {}
			for frase in frases:
				if len(frase) > 0 and frase[:1] != "#":
					contenido = frase.split(' ')
					try:
						data[contenido[0]] = float(contenido[2])
					except:
						if contenido[2] == "True":
							data[contenido[0]] = True
						elif contenido[2] == "False":
							data[contenido[0]] = False
						else:
							data[contenido[0]] = contenido[2]
		return data

class ExcelIO():
	"""
		Obtiene los datos y envia resultados en formato .xlsx
	"""

	def Input(self, file):
		#Importación del archivo
		xl_file = pd.read_excel(file)
		Data = pd.DataFrame(xl_file)

		#Procesamiento de NaN
		#Data = Data.replace(np.nan, 0)

		#Datos relevantes
		Relevant_Data = Data.iloc[:, 1:]
		return Relevant_Data

	def Output(self):
		pass

class ANOVA1(ExcelIO, TxtData):
	def __init__(self, **kwargs):
		#Lectura de Generalidades.txt
		txt_Dat = self.txtinput()
		print(txt_Dat)

		#Datos de interés
		Data = self.Input(kwargs['file'])
		Encabezados = list(Data)
		#Datos estadísticos generales
		General = self.EG(Data, Encabezados)

		#Eliminación de datos anómalos
		for columna in Encabezados:
			if General[columna]['Datos originales']['cant_datos'] <= 10 or \
				txt_Dat['Grubbs']:
				DATA = self.GrubbsAnomaly(General[columna], txt_Dat['alpha'])
			else:
				self.NormalDist(General[columna], kwargs['directory'], columna)
			if txt_Dat['NormalDist']:
				self.NormalDist(General[columna], kwargs['directory'], columna)

	def NormalDist(self, Data, directory, columna):
		Data['Datos originales']['Orden'] = np.searchsorted(
			np.sort(Data['Datos originales']['data']),
			Data['Datos originales']['data'], side='left')
		#Creación de formatos de datos "Fracción", "Z", Log(P) y %
		Data['Datos originales']['Fracción'] = np.zeros(
			Data['Datos originales']['cant_datos'])
		Data['Datos originales']['Z'] = np.zeros(
			Data['Datos originales']['cant_datos'])
		Data['Datos originales']['logP'] = np.zeros(
			Data['Datos originales']['cant_datos'])
		Data['Datos originales']['Porcentaje'] = np.zeros(
			Data['Datos originales']['cant_datos'])
		#Llenado de la información
		for i in range(Data['Datos originales']['cant_datos']):
			Data['Datos originales']['Orden'][i] += 1
			Data['Datos originales']['Fracción'][i] = \
			(Data['Datos originales']['Orden'][i]-0.5)/Data\
			['Datos originales']['cant_datos']
			Data['Datos originales']['Z'][i] = np.random.normal(
				Data['Datos originales']['Fracción'][i])
			Data['Datos originales']['logP'][i] = math.log10(
				100*Data['Datos originales']['Fracción'][i])
			Data['Datos originales']['Porcentaje'][i] = 100*\
			Data['Datos originales']['Fracción'][i]
		#GRÁFICAS
		#Prueba de distribución normal
		x = Data['Datos originales']['data']
		y = Data['Datos originales']['Fracción']
		fig, ax = plt.subplots()
		ax.plot(x,y,'.')
		#Línea de tendencia
		z = np.polyfit(x,y,1)
		p = np.poly1d(z)
		ax.plot(x,p(x), "r-")
		#Generalidades del gráfico
		plt.title("Prueba de distirbución normal - " + columna)
		plt.xlabel("Datos")
		plt.ylabel("Z")
		#plt.show()
		#Guardar gráfica
		plt.savefig(directory + columna + '.png')
		#print(Data)

	def GrubbsAnomaly(self, Data, alpha):
		col = {}
		col['data'] = grubbs.test(Data\
			['Datos originales']['data'], alpha=alpha)
		#Conteo de datos
		col['cant_datos'] = len(col['data'])

		Data['Datos definitivos'] = col
	
	def EG(self, Data, Encabezados):
		#Conversión de datos a matriz
		Matriz = Data.values

		#Organización de datos en diccionario
		Datos = {}
		con = 0
		for columna in Encabezados:
			col = {}
			#Conteo de datos por columna
			contador = 0
			for i in range(Matriz.shape[0]):
				if np.isnan(Matriz[i][con]):
					pass
				else:
					contador += 1
			dat = np.zeros(contador)
			
			#Reemplazar valores por reales en cada vector columna
			for i in range(contador):
				dat[i] = Matriz[i][con]
			
			col['data'] = dat
			col['cant_datos'] = contador
			col['promedio']	= np.mean(dat)
			col['std'] = np.std(dat)

			original = {}
			original['Datos originales'] = col

			Datos[columna] = original
			con += 1
		return Datos

if __name__:
	direc = 'Datos/Ejemplo1/'
	file = direc + 'Ejemplo1.xlsx'
	ANOVA1(file=file, alpha=0.05, directory=direc)
