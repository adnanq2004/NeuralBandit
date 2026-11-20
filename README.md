# NeuralBandit

Columbia COMS4726 Final Project - Implementation and "Improvement" of Neuralbandit algortihm, from “A Neural Networks Committee for the Contextual Bandit Problem” Robin Allesiardo · Raphaël Féraud · Djallel Bouneffouf





Running Information:



Quick test (Original):

&#x09;python neuralbandit3_original.py --steps 2000 --models small



Longer run (Original):

&#x09;python neuralbandit3_original.py --steps 50000 --models small



Concept-drift test (Original):

&#x09;python neuralbandit3_original.py --steps 20000 --drift-every 5000 --log-every 500 --output neuralbandit3_original_metrics_sigmoid.csv


&#x09;python neuralbandit3_original.py --steps 50000 --drift-every 5000 --log-every 500 --output neuralbandit3_original_metrics_sigmoid_more.csv



Concept-drift test (Activation Changes):

&#x09;python neuralbandit3_improvingactivation.py --steps 20000 --drift-every 5000 --log-every 500 --activation relu --output neuralbandit3_improvingactivation_metrics_relu.csv


&#x09;python neuralbandit3_improvingactivation.py --steps 20000 --drift-every 5000 --log-every 500 --activation tanh --output neuralbandit3_improvingactivation_metrics_tanh.csv


&#x09;python neuralbandit3_improvingactivation.py --steps 20000 --drift-every 5000 --log-every 500 --activation leaky_relu --output neuralbandit3_improvingactivation_metrics_leaky_relu.csv


&#x09;python neuralbandit3_improvingactivation.py --steps 50000 --drift-every 5000 --log-every 500 --activation relu --output neuralbandit3_improvingactivation_metrics_relu_more.csv


&#x09;python neuralbandit3_improvingactivation.py --steps 50000 --drift-every 5000 --log-every 500 --activation tanh --output neuralbandit3_improvingactivation_metrics_tanh_more.csv


&#x09;python neuralbandit3_improvingactivation.py --steps 50000 --drift-every 5000 --log-every 500 --activation leaky_relu --output neuralbandit3_improvingactivation_metrics_leaky_relu_more.csv



Exploration-rate test (Gamma, Gamma_model)

&#x09;python neuralbandit3_improvingactivation.py --steps 50000 --drift-every 5000 --log-every 500 --activation tanh --gamma 0.01 --gamma-model 0.1 --output neuralbandit3_improvingexploration_metrics1_.csv

&#x09;python neuralbandit3_improvingactivation.py --steps 50000 --drift-every 5000 --log-every 500 --activation tanh --gamma 0.05 --gamma-model 0.1 --output neuralbandit3_improvingexploration_metrics2_.csv

&#x09;python neuralbandit3_improvingactivation.py --steps 50000 --drift-every 5000 --log-every 500 --activation tanh --gamma 0.10 --gamma-model 0.1 --output neuralbandit3_improvingexploration_metrics3_.csv