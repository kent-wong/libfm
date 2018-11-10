// Copyright (C) 2010, 2011, 2012, 2013, 2014 Steffen Rendle
// Contact:   srendle@libfm.org, http://www.libfm.org/
//
// This file is part of libFM.
//
// libFM is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// libFM is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with libFM.  If not, see <http://www.gnu.org/licenses/>.
//
//
// fm_learn_sgd.h: Stochastic Gradient Descent based learning for
// classification and regression
//
// Based on the publication(s):
// - Steffen Rendle (2010): Factorization Machines, in Proceedings of the 10th
//   IEEE International Conference on Data Mining (ICDM 2010), Sydney,
//   Australia.

#ifndef FM_LEARN_SGD_ELEMENT_H_
#define FM_LEARN_SGD_ELEMENT_H_

#include "fm_learn_sgd.h"

class fm_learn_sgd_element: public fm_learn_sgd {
 public:
  virtual void init();
  virtual void learn(Data& train, Data& test);
};

// Implementation
void fm_learn_sgd_element::init() {
  fm_learn_sgd::init();

  if (log != NULL) {
    log->addField("rmse_train", std::numeric_limits<double>::quiet_NaN());
  }
}

void fm_learn_sgd_element::learn(Data& train, Data& test) {
	// wk_debug
	std::cout << "*** fm_learn_sgd_element::learn() ***" << std::endl;
	std::cout << "*** num_iter: " << num_iter << " ***" << std::endl;

  fm_learn_sgd::learn(train, test);

#ifdef ENABLE_MPI
	if (fm->my_rank == MPI_SERVER_NODE) {
		for (int i = 0; i < num_iter; i++) {
			for (int worker = 1; worker < fm->world_size; worker++)
				fm->server_pull();

			fm->server_learn(learn_rate);
			fm->server_push();

			double rmse_train = evaluate(train);
			double rmse_test = evaluate(test);
			std::cout << "Server: #Iter=" << std::setw(3) << i << "\tTrain=" << rmse_train << "\tTest=" << rmse_test << std::endl;
		}
		return ;
	}
#endif

  std::cout << "SGD: DON'T FORGET TO SHUFFLE THE ROWS IN TRAINING DATA TO GET THE BEST RESULTS." << std::endl;
  // SGD
  for (int i = 0; i < num_iter; i++) {

    double iteration_time = getusertime();
    for (train.data->begin(); !train.data->end(); train.data->next()) {
      double p = fm->predict(train.data->getRow(), sum, sum_sqr);
      double mult = 0;
      if (task == 0) {
        p = std::min(max_target, p);
        p = std::max(min_target, p);
        mult = -(train.target(train.data->getRowIndex())-p);
      } else if (task == 1) {
        mult = -train.target(train.data->getRowIndex())*(1.0-1.0/(1.0+exp(-train.target(train.data->getRowIndex())*p)));
      }
      SGD(train.data->getRow(), mult, sum);
    }
    iteration_time = (getusertime() - iteration_time);
    //double rmse_train = evaluate(train);
    //double rmse_test = evaluate(test);
    //std::cout << "#Iter=" << std::setw(3) << i << "\tTrain=" << rmse_train << "\tTest=" << rmse_test << std::endl;
    if (log != NULL) {
      double rmse_train = evaluate(train);
      log->log("rmse_train", rmse_train);
      log->log("time_learn", iteration_time);
      log->newLine();
    }
#ifdef ENABLE_MPI
	if (fm->my_rank != MPI_SERVER_NODE) {
		// worker node
		fm->worker_push();	
		fm->worker_pull();

		double rmse_train = evaluate(train);
		double rmse_test = evaluate(test);
		std::cout << "Client: #Iter=" << std::setw(3) << i << "\tTrain=" << rmse_train << "\tTest=" << rmse_test << std::endl;
	}
#endif
  }
}

#endif /*FM_LEARN_SGD_ELEMENT_H_*/
