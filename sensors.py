# coding: utf8

import api

from leto.unstructured.sensorial.actions import ActionSensor, ActionPipelineSensor
from leto.unstructured.sensorial.entities import EntitySensor
from leto.unstructured.sensorial.coreferences import CoreferenceSensor
from leto.unstructured.sensorial.sentiment_sam import SentimentSensor
from leto.unstructured.sensorial.project.aspects import AspectsSensor
from leto.unstructured.sensorial.project.aspects_crf import AspectsCRF
# from leto.unstructured.sensorial.project.aspects_evaluation import AspectsSensor
from leto.unstructured.sensorial.project.aspects_handL import AspectsSensor



@api.command(name="Extract Actions", section="Basic Sensors")
def extract_actions(text: str):
	action_sensor = ActionSensor()

	return {
		"text": text,
		"actions": list(action_sensor.run(text))
	}


@api.command(name="Extract Entities", section="Basic Sensors")
def extract_entities(text: str):
	entity_sensor = EntitySensor()

	return {
		"text": text,
		"entities": list(entity_sensor.run(text))
	}


@api.command(name="Extract Coreferences", section="Basic Sensors")
def extract_corefences(text: str):
	coref_sensor = CoreferenceSensor()

	return {
		"text": text,
		"coreferences": list(coref_sensor.run(text))
	}


@api.command(name="Extract Emotions", section="Basic Sensors")
def extract_sentiments(text: str):
	sent_sensor = SentimentSensor()

	return {
		"text": text,
		"emotions": sent_sensor.run(text)
	}


@api.command(name="Extract Actions Pipeline (actions + entities + coreferences)", section="Basic Sensors")
def extract_actions_pipeline(text: str):
	action_entity_sensor = ActionPipelineSensor()
	actions = list(action_entity_sensor.run(text))

	return {
		"text": text,
		"actions": actions,
	}

@api.command(name="Extract Aspects based in frequency", section="Basic Sensors")
def extract_aspects_pipeline(text: str):
	aspect_entity_sensor = AspectsSensor(text)
	aspects = list(aspect_entity_sensor.FREQ(1))

	return {
		"text": text,
		"aspects": aspects,
	}



@api.command(name="Extract Aspects based in frequency using prunning and expansion", section="Basic Sensors")
def extract_aspectsHL_pipeline(text: str):
	aspect_entity_sensor = AspectsSensor(text)
	aspects = list(aspect_entity_sensor.FrecuencyBasedHL())

	return {
		"text": text,
		"aspects": aspects,
	}

@api.command(name="Extract Aspects using Conditional Random Field", section="Basic Sensors")
def extract_aspects_crf_pipeline(text: str):
	path_train = "./data/corpus_train.txt"
	path_label= "./data/corpus_test_aspect.txt"
	path_label_train = "./data/corpus_train_aspect.txt"
	aspect_entity_sensor = AspectsCRF(text,path_train,path_label, path_label_train)
	aspects = aspect_entity_sensor.aspects

	return {
		"text": text,
		"aspects": aspects,
	}

