#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 300;
  weights.resize(num_particles, 1.0);

  // Create normal distrobutions
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // Initialize the particles
  for(int i=0; i<num_particles; ++i){
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    particles.push_back(particle);    
  }

  // Is initialized flag
  is_initialized = true;
  return;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  for(auto& p: particles){
    if(fabs(yaw_rate) > 0.001){
      p.x += velocity/yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
      p.y += velocity/yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
      p.theta += yaw_rate*delta_t;
    }else{
      p.x += velocity*delta_t*cos(p.theta);
      p.y += velocity*delta_t*sin(p.theta);
    }

    normal_distribution<double> dist_x(p.x, std_pos[0]);
    normal_distribution<double> dist_y(p.y, std_pos[1]);
    normal_distribution<double> dist_theta(p.theta, std_pos[2]);

    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
  }
  return;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
}

vector<LandmarkObs> associate_data(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations){
  vector<LandmarkObs> associated_landmarks;
  LandmarkObs closest;

  for(auto obs: observations){
    double shortest = 1E10;
    for(auto pred: predicted){
      double distance = dist(obs.x, obs.y, pred.x, pred.y);
      if(distance < shortest){
	shortest = distance;
	closest = pred;
      }
    }
    associated_landmarks.push_back(closest);
  }
  return associated_landmarks;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  double sigma_x = std_landmark[0];
  double sigma_y = std_landmark[1];

  for(int i=0; i<particles.size(); ++i){
    // Collect all landmarks within sensor range of current particle
    Particle p = particles[i];

    // Transform observations to MAP system
    vector<LandmarkObs> transformed_observations;
    for(auto observation: observations){
      LandmarkObs transformed_observation;
      transformed_observation.x = p.x + observation.x*cos(p.theta) - observation.y*sin(p.theta);
      transformed_observation.y = p.y + observation.x*sin(p.theta) + observation.y*cos(p.theta);
      transformed_observation.id = observation.id;
      transformed_observations.push_back(transformed_observation);
    }

    // Get landmarks within sight.
    vector<LandmarkObs> predicted;
    for(auto landmark: map_landmarks.landmark_list){
      double distance = dist(p.x, p.y, landmark.x_f, landmark.y_f);
      if(distance < sensor_range){
	LandmarkObs current_landmark;
	current_landmark.id = landmark.id_i;
	current_landmark.x = landmark.x_f;
	current_landmark.y = landmark.y_f;
	predicted.push_back(current_landmark);
      }
    }

    // Associate the nearest landmark to every observation
    vector<LandmarkObs> associated_landmarks;
    associated_landmarks = associate_data(predicted, transformed_observations);

    double probab = 1;
    for(int j=0; j<associated_landmarks.size(); ++j){
      double dx = transformed_observations.at(j).x - associated_landmarks.at(j).x;
      double dy = transformed_observations.at(j).y - associated_landmarks.at(j).y;
      probab *= 1.0/(2*M_PI*sigma_x*sigma_y) * exp(-dx*dx / (2*sigma_x*sigma_x)) * exp(-dy*dy / (2*sigma_y*sigma_y));
    }

    p.weight = probab;
    weights[i] = probab;
  }

  return;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  discrete_distribution<int> d(weights.begin(), weights.end());
  vector<Particle> weighted_sample(num_particles);
  for(int i=0; i<num_particles; ++i){
    int j = d(gen);
    weighted_sample.at(i) = particles.at(j);
  }
  particles = weighted_sample;
  return;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

  // Clear previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
