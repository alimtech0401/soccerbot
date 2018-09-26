/**
  *****************************************************************************
  * @file    UartDriver.cpp
  * @author  Tyler Gamvrelis
  * @brief   TODO -- brief description of file
  *
  * @defgroup UartDriver
  * @brief    TODO
  * @{
  *****************************************************************************
  */




/********************************* Includes **********************************/
#include "UartDriver.h"




namespace UART{
/********************************* UartDriver ********************************/
// Public
// ----------------------------------------------------------------------------
UartDriver::UartDriver(){

}

void UartDriver::setUartInterface(UartInterface* hw_if){
    this->hw_if = hw_if;
}

#ifdef THREADED
void UartDriver::setOSInterface(FreeRTOSInterface* os_if){
    this->os_if = os_if;
}
#endif

void UartDriver::setUartPtr(UART_HandleTypeDef* uartHandlePtr){
    hw_if->setUartPtr(uartHandlePtr);
}

void UartDriver::setIOType(IO_Type io_type){
    this->io_type = io_type;
}

IO_Type UartDriver::getIOType(void) const{
    return this->io_type;
}

bool UartDriver::transmit(
    uint8_t* arrTransmit,
    size_t numBytes
)
{
#if defined(THREADED)
    uint32_t notification = 0;
    BaseType_t status = pdFALSE;
#endif
    bool retval = false;

    if(hw_if != nullptr){
        switch(io_type) {
#if defined(THREADED)
            case IO_Type::DMA:
                if(os_if != nullptr){
                    if(hw_if->transmitDMA(arrTransmit, numBytes) == HAL_OK){
                        status = os_if->OS_xTaskNotifyWait(0, NOTIFIED_FROM_TX_ISR, &notification, MAX_BLOCK_TIME);

                        if(status != pdTRUE || !CHECK_NOTIFICATION(notification, NOTIFIED_FROM_TX_ISR)){
                            retval = false;
                        }
                    }
                    else{
                        retval = false;
                    }
                }
                break;
            case IO_Type::IT:
                if(os_if != nullptr){
                    if(hw_if->transmitIT(arrTransmit, numBytes) == HAL_OK){
                        status = os_if->OS_xTaskNotifyWait(0, NOTIFIED_FROM_TX_ISR, &notification, MAX_BLOCK_TIME);

                        if(status != pdTRUE || !CHECK_NOTIFICATION(notification, NOTIFIED_FROM_TX_ISR)){
                            retval = false;
                        }
                    }
                    else{
                        retval = false;
                    }
                }
                break;
#endif
            case IO_Type::POLL:
            default:
                retval = (hw_if->transmitPoll(arrTransmit, numBytes, POLLED_TRANSFER_TIMEOUT) == HAL_OK);
                break;
        }

        if(retval != HAL_OK){
            hw_if->abortTransmit();
        }
    }

    return retval;
}

bool UartDriver::receive(
    uint8_t* arrReceive,
    size_t numBytes
)
{
#if defined(THREADED)
    uint32_t notification = 0;
    BaseType_t status = pdFALSE;
#endif
    bool retval = false;

    if(hw_if != nullptr){
        switch(io_type) {
#if defined(THREADED)
            case IO_Type::DMA:
                if(os_if != nullptr){
                    if(hw_if->receiveDMA(arrReceive, numBytes) == HAL_OK){
                        status = os_if->OS_xTaskNotifyWait(0, NOTIFIED_FROM_RX_ISR, &notification, MAX_BLOCK_TIME);

                        if(status != pdTRUE || !CHECK_NOTIFICATION(notification, NOTIFIED_FROM_RX_ISR)){
                            retval = false;
                        }
                    }
                    else{
                        retval = false;
                    }
                }
                break;
            case IO_Type::IT:
                if(os_if != nullptr){
                    if(hw_if->receiveIT(arrReceive, numBytes) == HAL_OK){
                        status = os_if->OS_xTaskNotifyWait(0, NOTIFIED_FROM_RX_ISR, &notification, MAX_BLOCK_TIME);

                        if(status != pdTRUE || !CHECK_NOTIFICATION(notification, NOTIFIED_FROM_RX_ISR)){
                            retval = false;
                        }
                    }
                    else{
                        retval = false;
                    }
                }
                break;
#endif
            case IO_Type::POLL:
            default:
                retval = (hw_if->receivePoll(arrReceive, numBytes, POLLED_TRANSFER_TIMEOUT) == HAL_OK);
                break;
        }

        if(retval != HAL_OK){
            hw_if->abortReceive();
        }
    }

    return retval;
}

} // end namespace uart



/**
 * @}
 */
/* end - module name */
