/**
  *****************************************************************************
  * @file    example_cpp.h
  * @author  TODO -- your name here
  * @brief   TODO -- briefly describe this file (if necessary)
  *****************************************************************************
  */




/******************** Define to prevent recursive inclusion ******************/
#ifndef __EXAMPLE_CPP_H__
#define __EXAMPLE_CPP_H__




/********************************* Includes **********************************/
#include <stdint.h>




/********************************* Namespace *********************************/
/**
 * @addtogroup Namespace
 * @brief Brief namespace description
 * @ingroup Parent_Module_Name
 * @{
 */
namespace Namespace {




/********************************** Macros ***********************************/




/********************************* Constants *********************************/




/********************************** Classes **********************************/
/** 
 * @class   Class
 * @brief   Short class description
 * @details Detailed class description
 */
class Class {
public:
    Class();
   ~Class();

    /** 
     * @brief  Member description here
     * @param  a description of the significance of a
     * @param  s description of the significance of s
     * @return The test results
     */
    int testMe(int a, const char* s);

protected:

private:

}; // class Class

} // namespace Namespace

/**
 * @}
 */
/* end - Namespace */

#endif /* __EXAMPLE_CPP_H__ */
