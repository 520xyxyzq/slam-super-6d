/**
 * @file SmartMaxMixtureFactor.h
 * @brief Max-Mixture factor providing several extra interfaces for weight
 * updates and association retrieval
 * @author Kevin Doherty, kdoherty@mit.edu
 *
 * Copyright 2021 The Ambitious Folks of the MRG
 */

#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Symbol.h>
#include <math.h>

#include <algorithm>
#include <limits>
#include <vector>

/**
 * @brief GTSAM implementation of a max-mixture factor
 *
 * r(x) = min_i -log(w_i) + r_i(x)
 *
 * The error returned from this factor is the minimum error + weight
 * over all of the component factors
 * See Olson and Agarwal RSS 2012 for details
 */
template <class T>
class MaxMixtureFactor : public gtsam::NonlinearFactor {
 private:
  std::vector<T> factors_;
  std::vector<double> log_weights_;

 public:
  using Base = gtsam::NonlinearFactor;

  MaxMixtureFactor() = default;

  explicit MaxMixtureFactor(const std::vector<gtsam::Key> keys,
                            const std::vector<T> factors,
                            const std::vector<double> weights)
      : Base(keys) {
    assert((weights.size() == factors.size()) &&
           "MaxMixFactor: factors weights dimension mismatch");
    factors_ = factors;
    for (int i = 0; i < weights.size(); i++) {
      log_weights_.push_back(log(weights[i]));
    }
  }

  MaxMixtureFactor& operator=(const MaxMixtureFactor& rhs) {
    Base::operator=(rhs);
    this->factors_ = rhs.factors_;
    this->log_weights_ = rhs.log_weights_;
  }

  virtual ~MaxMixtureFactor() = default;

  double error(const gtsam::Values& values) const override {
    double min_error = std::numeric_limits<double>::infinity();
    for (int i = 0; i < factors_.size(); i++) {
      double error = factors_[i].error(values) - log_weights_[i];
      if (error < min_error) {
        min_error = error;
      }
    }
    return min_error;
  }

  size_t dim() const override {
    if (factors_.size() > 0) {
      return factors_[0].dim();
    } else {
      return 0;
    }
  }

  boost::shared_ptr<gtsam::GaussianFactor> linearize(
      const gtsam::Values& x) const override {
    double min_error = std::numeric_limits<double>::infinity();
    int idx_min = -1;
    for (int i = 0; i < factors_.size(); i++) {
      double error = factors_[i].error(x) - log_weights_[i];
      if (error < min_error) {
        min_error = error;
        idx_min = i;
      }
    }
    return factors_[idx_min].linearize(x);
  }

  gtsam::FastVector<gtsam::Key> getAssociationKeys(
      const gtsam::Values& x) const {
    double min_error = std::numeric_limits<double>::infinity();
    int idx_min = -1;
    for (int i = 0; i < factors_.size(); i++) {
      double error = factors_[i].error(x) - log_weights_[i];
      if (error < min_error) {
        min_error = error;
        idx_min = i;
      }
    }
    return factors_[idx_min].keys();
  }

  void updateWeights(const std::vector<double>& weights) {
    if (weights.size() != log_weights_.size()) {
      std::cerr << "Attempted to update weights with incorrectly sized vector."
                << std::endl;
      return;
    }
    for (int i = 0; i < weights.size(); i++) {
      log_weights_[i] = log(weights[i]);
    }
  }

  /* size_t dim() const override { return 1; } */
};
