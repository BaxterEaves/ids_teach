// IDSTeach: Generate data to teach continuous categorical data.
// Copyright (C) 2015  Baxter S. Eaves Jr.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef idsteach_cxx_debug_guard
#define idsteach_cxx_debug_guard

    // TODO: refactor some of this redundant code and add common exceptions
    /* Custom assert macros.
    The goal of these macros is to provide more information than the standard cassert macros and to
    bypass cython killing casserts.
    */

    #include <cmath>
    #include <iostream>
    #include <sstream>

    // Exception class for when an iterative algroithm fails to complete within the maximum allowed
    // number of iterations.
    class MaxIterationsReached {
    public:
        MaxIterationsReached(const size_t &iters) : _iters(iters){};
        const char* what(){
            std::ostringstream msg;
            msg << "Maximum number of iterations (" << _iters << ") reached.\n";
            return msg.str().c_str();
        }
    private:
        size_t _iters;
    };

    #ifndef NIDSDEBUG
        // #define PRINT_DEBUG_MESSAGE(os, msg)                                    \
        //     do{                                                                 \
        //         (os) << "DEBUG: " << __FILE__ << "(" << __LINE__ << ") in ";    \
        //         (os) << __PRETTY_FUNCTION__ << std::endl;                       \
        //         (os) << "----" << msg << std::endl                              \
        //     }while(0)

        #define ASSERT( os, test )                                              \
            do{                                                                 \
            if (!(test)){                                                       \
                (os) << "DEBUG: " << __FILE__ << "(" << __LINE__ << ") in ";    \
                (os) << __PRETTY_FUNCTION__ << std::endl;                       \
                (os) << "\t" << "TEST: " << #test << " failed." << std::endl;   \
                throw 0;                                                        \
            }                                                                   \
            }while(0)

        #define ASSERT_INFO( os, msg, test )                                \
            do{                                                             \
            if (!(test)){                                                   \
                (os) << "DEBUG: " << __FILE__ << "(" << __LINE__ << ") in ";\
                (os) << __PRETTY_FUNCTION__ << msg << std::endl;            \
                (os) << "\t" << "TEST: " << #test << std::endl;             \
                throw 0;                                                    \
            }                                                               \
            }while(0)

        #define ASSERT_GREATER_THAN_ZERO(os, number)                                        \
            do{                                                                             \
                if(number <= 0){                                                            \
                    (os) << "DEBUG: " << __FILE__ << "(" << __LINE__ << ") in ";            \
                    (os) << __PRETTY_FUNCTION__ << #number;                                 \
                    (os) << "(" << number << ") should be greater than zero." << std::endl; \
                    throw 0;                                                                \
                }                                                                           \
            }while(0)

        #define ASSERT_EQUAL(os, X, Y)                                                      \
            do{                                                                             \
                if(X != Y){                                                                 \
                    (os) << "DEBUG: " << __FILE__ << "(" << __LINE__ << ") in ";            \
                    (os) << __PRETTY_FUNCTION__ << #X;                                      \
                    (os) << "(" << X << ") should be equal to " << #Y ;                     \
                    (os) << "(" << Y << ")." << std::endl;                                  \
                    throw 0;                                                                \
                }                                                                           \
            }while(0)

        #define ASSERT_NOT_EQUAL(os, X, Y)                                                  \
            do{                                                                             \
                if(X == Y){                                                                 \
                    (os) << "DEBUG: " << __FILE__ << "(" << __LINE__ << ") in ";            \
                    (os) << __PRETTY_FUNCTION__ << #X;                                      \
                    (os) << "(" << X << ") should not be equal to " << #Y ;                 \
                    (os) << "(" << Y << ")." << std::endl;                                  \
                    throw 0;                                                                \
                }                                                                           \
            }while(0)

        #define ASSERT_IS_A_NUMBER(os, number)                                      \
            do{                                                                     \
                if(std::isnan(number) or std::isinf(number)){                       \
                    (os) << "DEBUG: " << __FILE__ << "(" << __LINE__ << ") in ";    \
                    (os) << __PRETTY_FUNCTION__ << #number;                         \
                    (os) << "(" << number << ") is Inf or NaN." << std::endl;       \
                    throw 0;                                                        \
                }                                                                   \
            }while(0)

        #define DEBUG_MESSAGE(os, msg)                                                          \
            do{                                                                                 \
                (os) << "DEBUG: " << __FILE__ << "(" << __LINE__ << ") " << msg << std::endl;   \
                throw 0;                                                                        \
            }while(0)
    #else
        #define ASSERT( os, test )
        #define ASSERT_INFO( os, msg, test )
        #define ASSERT_GREATER_THAN_ZERO(os, number)
        #define ASSERT_IS_A_NUMBER(os, number)
        #define ASSERT_EQUAL(os, X, Y)
        #define ASSERT_NOT_EQUAL(os, X, Y)
        #define DEBUG_MESSAGE(os, msg)
    #endif

#endif
