/**
  ******************************************************************************
  * @file    freertos.c
  * @brief   Code for freertos application
  * @author  Gokul
  * @author  Tyler
  * @author  Izaak
  *
  * @defgroup FreeRTOS FreeRTOS
  * @brief    Everything related to FreeRTOS
  ******************************************************************************
  * This notice applies to any and all portions of this file
  * that are not between comment pairs USER CODE BEGIN and
  * USER CODE END. Other portions of this file, whether 
  * inserted by the user or by software development tools
  * are owned by their respective copyright owners.
  *
  * Copyright (c) 2018 STMicroelectronics International N.V. 
  * All rights reserved.
  *
  * Redistribution and use in source and binary forms, with or without 
  * modification, are permitted, provided that the following conditions are met:
  *
  * 1. Redistribution of source code must retain the above copyright notice, 
  *    this list of conditions and the following disclaimer.
  * 2. Redistributions in binary form must reproduce the above copyright notice,
  *    this list of conditions and the following disclaimer in the documentation
  *    and/or other materials provided with the distribution.
  * 3. Neither the name of STMicroelectronics nor the names of other 
  *    contributors to this software may be used to endorse or promote products 
  *    derived from this software without specific written permission.
  * 4. This software, including modifications and/or derivative works of this 
  *    software, must execute solely and exclusively on microcontroller or
  *    microprocessor devices manufactured by or for STMicroelectronics.
  * 5. Redistribution and use of this software other than as permitted under 
  *    this license is void and will automatically terminate your rights under 
  *    this license. 
  *
  * THIS SOFTWARE IS PROVIDED BY STMICROELECTRONICS AND CONTRIBUTORS "AS IS" 
  * AND ANY EXPRESS, IMPLIED OR STATUTORY WARRANTIES, INCLUDING, BUT NOT 
  * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
  * PARTICULAR PURPOSE AND NON-INFRINGEMENT OF THIRD PARTY INTELLECTUAL PROPERTY
  * RIGHTS ARE DISCLAIMED TO THE FULLEST EXTENT PERMITTED BY LAW. IN NO EVENT 
  * SHALL STMICROELECTRONICS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
  * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
  * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
  * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "FreeRTOS.h"
#include "task.h"
#include "cmsis_os.h"

/* USER CODE BEGIN Includes */     
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "stm32f4xx_hal.h"
#include "usart.h"
#include "gpio.h"
#include "i2c.h"
#include "../Drivers/MPU6050/MPU6050.h"
#include "sharedMacros.h"
#include "UART_Handler.h"
#include "../Drivers/Dynamixel/DynamixelProtocolV1.h"
#include "../Drivers/Communication/Communication.h"

/* USER CODE END Includes */

/* Variables -----------------------------------------------------------------*/
osThreadId defaultTaskHandle;
uint32_t defaultTaskBuffer[ 512 ];
osStaticThreadDef_t defaultTaskControlBlock;
osThreadId UART1_Handle;
uint32_t UART1_Buffer[ 128 ];
osStaticThreadDef_t UART1_ControlBlock;
osThreadId UART2_Handle;
uint32_t UART2_Buffer[ 128 ];
osStaticThreadDef_t UART2_ControlBlock;
osThreadId UART3_Handle;
uint32_t UART3_Buffer[ 128 ];
osStaticThreadDef_t UART3_ControlBlock;
osThreadId UART4_Handle;
uint32_t UART4_Buffer[ 128 ];
osStaticThreadDef_t UART4_ControlBlock;
osThreadId UART6_Handle;
uint32_t UART6_Buffer[ 128 ];
osStaticThreadDef_t UART6_ControlBlock;
osThreadId IMUTaskHandle;
uint32_t IMUTaskBuffer[ 512 ];
osStaticThreadDef_t IMUTaskControlBlock;
osThreadId rxTaskHandle;
uint32_t rxTaskBuffer[ 512 ];
osStaticThreadDef_t rxTaskControlBlock;
osThreadId txTaskHandle;
uint32_t txTaskBuffer[ 512 ];
osStaticThreadDef_t txTaskControlBlock;
osMessageQId UART1_reqHandle;
uint8_t UART1_reqBuffer[ 16 * sizeof( UARTcmd_t ) ];
osStaticMessageQDef_t UART1_reqControlBlock;
osMessageQId UART2_reqHandle;
uint8_t UART2_reqBuffer[ 16 * sizeof( UARTcmd_t ) ];
osStaticMessageQDef_t UART2_reqControlBlock;
osMessageQId UART3_reqHandle;
uint8_t UART3_reqBuffer[ 16 * sizeof( UARTcmd_t ) ];
osStaticMessageQDef_t UART3_reqControlBlock;
osMessageQId UART4_reqHandle;
uint8_t UART4_reqBuffer[ 16 * sizeof( UARTcmd_t ) ];
osStaticMessageQDef_t UART4_reqControlBlock;
osMessageQId UART6_reqHandle;
osMessageQId TXQueueHandle;
uint8_t TXQueueBuffer[ 32 * sizeof( IMUStruct ) ];
osStaticMessageQDef_t TXQueueControlBlock;
uint8_t UART6_reqBuffer[ 16 * sizeof( UARTcmd_t ) ];
osStaticMessageQDef_t UART6_reqControlBlock;
osMessageQId UART_rxHandle;
uint8_t UART_rxBuffer[ 32 * sizeof( UARTcmd_t ) ];
osStaticMessageQDef_t UART_rxControlBlock;
osMutexId PCUARTHandle;
osStaticMutexDef_t PCUARTControlBlock;

/* USER CODE BEGIN Variables */

enum motorNames {MOTOR1, MOTOR2, MOTOR3, MOTOR4, MOTOR5,
				 MOTOR6, MOTOR7, MOTOR8, MOTOR9, MOTOR10,
				 MOTOR11, MOTOR12, MOTOR13, MOTOR14, MOTOR15,
				 MOTOR16, MOTOR17, MOTOR18
};

IMUnamespace::MPU6050 IMUdata (1, &hi2c1);

Dynamixel_HandleTypeDef Motor1, Motor2, Motor3 ,Motor4, Motor5,
						Motor6, Motor7, Motor8, Motor9, Motor10,
						Motor11, Motor12, Motor13, Motor14, Motor15,
						Motor16, Motor17, Motor18;


bool setupIsDone = false;
static volatile uint32_t error;
/* USER CODE END Variables */

/* Function prototypes -------------------------------------------------------*/
void StartDefaultTask(void const * argument);
void UART1_Handler(void const * argument);
void UART2_Handler(void const * argument);
void UART3_Handler(void const * argument);
void UART4_Handler(void const * argument);
void UART6_Handler(void const * argument);
void StartIMUTask(void const * argument);
void StartRxTask(void const * argument);
void StartTxTask(void const * argument);

void MX_FREERTOS_Init(void); /* (MISRA C 2004 rule 8.1) */

/* USER CODE BEGIN FunctionPrototypes */

/* USER CODE END FunctionPrototypes */

/* GetIdleTaskMemory prototype (linked to static allocation support) */
#ifdef __cplusplus
extern "C"{
#endif

void vApplicationGetIdleTaskMemory( StaticTask_t **ppxIdleTaskTCBBuffer, StackType_t **ppxIdleTaskStackBuffer, uint32_t *pulIdleTaskStackSize );

#ifdef __cplusplus
}
#endif

/* Hook prototypes */

/* USER CODE BEGIN GET_IDLE_TASK_MEMORY */
static StaticTask_t xIdleTaskTCBBuffer;
static StackType_t xIdleStack[configMINIMAL_STACK_SIZE];
  
void vApplicationGetIdleTaskMemory( StaticTask_t **ppxIdleTaskTCBBuffer, StackType_t **ppxIdleTaskStackBuffer, uint32_t *pulIdleTaskStackSize )
{
  *ppxIdleTaskTCBBuffer = &xIdleTaskTCBBuffer;
  *ppxIdleTaskStackBuffer = &xIdleStack[0];
  *pulIdleTaskStackSize = configMINIMAL_STACK_SIZE;
  /* place for user code */
}                   
/* USER CODE END GET_IDLE_TASK_MEMORY */

/* Init FreeRTOS */

void MX_FREERTOS_Init(void) {
  /* USER CODE BEGIN Init */
       
  /* USER CODE END Init */

  /* Create the mutex(es) */
  /* definition and creation of PCUART */
  osMutexStaticDef(PCUART, &PCUARTControlBlock);
  PCUARTHandle = osMutexCreate(osMutex(PCUART));

  /* USER CODE BEGIN RTOS_MUTEX */
  /* add mutexes, ... */
  /* USER CODE END RTOS_MUTEX */

  /* USER CODE BEGIN RTOS_SEMAPHORES */
  /* add semaphores, ... */
  /* USER CODE END RTOS_SEMAPHORES */

  /* USER CODE BEGIN RTOS_TIMERS */
  /* start timers, add new ones, ... */
  /* USER CODE END RTOS_TIMERS */

  /* Create the thread(s) */
  /* definition and creation of defaultTask */
  osThreadStaticDef(defaultTask, StartDefaultTask, osPriorityIdle, 0, 512, defaultTaskBuffer, &defaultTaskControlBlock);
  defaultTaskHandle = osThreadCreate(osThread(defaultTask), NULL);

  /* definition and creation of UART1_ */
  osThreadStaticDef(UART1_, UART1_Handler, osPriorityNormal, 0, 128, UART1_Buffer, &UART1_ControlBlock);
  UART1_Handle = osThreadCreate(osThread(UART1_), NULL);

  /* definition and creation of UART2_ */
  osThreadStaticDef(UART2_, UART2_Handler, osPriorityNormal, 0, 128, UART2_Buffer, &UART2_ControlBlock);
  UART2_Handle = osThreadCreate(osThread(UART2_), NULL);

  /* definition and creation of UART3_ */
  osThreadStaticDef(UART3_, UART3_Handler, osPriorityNormal, 0, 128, UART3_Buffer, &UART3_ControlBlock);
  UART3_Handle = osThreadCreate(osThread(UART3_), NULL);

  /* definition and creation of UART4_ */
  osThreadStaticDef(UART4_, UART4_Handler, osPriorityNormal, 0, 128, UART4_Buffer, &UART4_ControlBlock);
  UART4_Handle = osThreadCreate(osThread(UART4_), NULL);

  /* definition and creation of UART6_ */
  osThreadStaticDef(UART6_, UART6_Handler, osPriorityNormal, 0, 128, UART6_Buffer, &UART6_ControlBlock);
  UART6_Handle = osThreadCreate(osThread(UART6_), NULL);

  /* definition and creation of IMUTask */
  osThreadStaticDef(IMUTask, StartIMUTask, osPriorityNormal, 0, 512, IMUTaskBuffer, &IMUTaskControlBlock);
  IMUTaskHandle = osThreadCreate(osThread(IMUTask), NULL);

  /* definition and creation of rxTask */
  osThreadStaticDef(rxTask, StartRxTask, osPriorityRealtime, 0, 512, rxTaskBuffer, &rxTaskControlBlock);
  rxTaskHandle = osThreadCreate(osThread(rxTask), NULL);

  /* definition and creation of txTask */
  osThreadStaticDef(txTask, StartTxTask, osPriorityHigh, 0, 512, txTaskBuffer, &txTaskControlBlock);
  txTaskHandle = osThreadCreate(osThread(txTask), NULL);

  /* USER CODE BEGIN RTOS_THREADS */
  /* add threads, ... */
  /* USER CODE END RTOS_THREADS */

  /* Create the queue(s) */
  /* definition and creation of UART1_req */
  osMessageQStaticDef(UART1_req, 16, UARTcmd_t, UART1_reqBuffer, &UART1_reqControlBlock);
  UART1_reqHandle = osMessageCreate(osMessageQ(UART1_req), NULL);

  /* definition and creation of UART2_req */
  osMessageQStaticDef(UART2_req, 16, UARTcmd_t, UART2_reqBuffer, &UART2_reqControlBlock);
  UART2_reqHandle = osMessageCreate(osMessageQ(UART2_req), NULL);

  /* definition and creation of UART3_req */
  osMessageQStaticDef(UART3_req, 16, UARTcmd_t, UART3_reqBuffer, &UART3_reqControlBlock);
  UART3_reqHandle = osMessageCreate(osMessageQ(UART3_req), NULL);

  /* definition and creation of UART4_req */
  osMessageQStaticDef(UART4_req, 16, UARTcmd_t, UART4_reqBuffer, &UART4_reqControlBlock);
  UART4_reqHandle = osMessageCreate(osMessageQ(UART4_req), NULL);

  /* definition and creation of UART6_req */
  osMessageQStaticDef(UART6_req, 16, UARTcmd_t, UART6_reqBuffer, &UART6_reqControlBlock);
  UART6_reqHandle = osMessageCreate(osMessageQ(UART6_req), NULL);

  /* definition and creation of UART_rx */
  osMessageQStaticDef(UART_rx, 32, UARTcmd_t, UART_rxBuffer, &UART_rxControlBlock);
  UART_rxHandle = osMessageCreate(osMessageQ(UART_rx), NULL);

  /* definition and creation of TXQueue */
  osMessageQStaticDef(TXQueue, 32, TXData_t, TXQueueBuffer, &TXQueueControlBlock);
  TXQueueHandle = osMessageCreate(osMessageQ(TXQueue), NULL);

  /* USER CODE BEGIN RTOS_QUEUES */
  /* add queues, ... */
  /* USER CODE END RTOS_QUEUES */
}

/**
 * @defgroup Threads Threads
 * @brief    These are functions run in the context of their own FreeRTOS
 *           threads
 *
 * @ingroup  FreeRTOS
 */

/* StartDefaultTask function */
/**
  * @brief  This function is executed in the context of the defaultTask
  *         thread. It initializes all data structures and peripheral
  *         devices associated with the application, and then assumes
  *         responsibility for distributing commands to the actuators
  *
  *         This function never returns.
  *
  * @ingroup Threads
  */
void StartDefaultTask(void const * argument)
{

    /* USER CODE BEGIN StartDefaultTask */
    Dynamixel_SetIOType(IO_POLL); // Configure IO

    Dynamixel_Init(&Motor12, 12, &huart6, GPIOC, GPIO_PIN_8, MX28TYPE);
    Dynamixel_Init(&Motor11, 11, &huart6, GPIOC, GPIO_PIN_8, MX28TYPE);
    Dynamixel_Init(&Motor10, 10, &huart6, GPIOC, GPIO_PIN_8, MX28TYPE);
    Dynamixel_Init(&Motor9, 9, &huart1, GPIOA, GPIO_PIN_8, MX28TYPE);
    Dynamixel_Init(&Motor8, 8, &huart1, GPIOA, GPIO_PIN_8, MX28TYPE);
    Dynamixel_Init(&Motor7, 7, &huart1, GPIOA, GPIO_PIN_8, MX28TYPE);
    Dynamixel_Init(&Motor6, 6, &huart4, GPIOC, GPIO_PIN_3, MX28TYPE);
    Dynamixel_Init(&Motor5, 5, &huart4, GPIOC, GPIO_PIN_3, MX28TYPE);
    Dynamixel_Init(&Motor4, 4, &huart4, GPIOC, GPIO_PIN_3, MX28TYPE);
    Dynamixel_Init(&Motor3, 3, &huart2, GPIOA, GPIO_PIN_4, MX28TYPE);
    Dynamixel_Init(&Motor2, 2, &huart2, GPIOA, GPIO_PIN_4, MX28TYPE);
    Dynamixel_Init(&Motor1, 1, &huart2, GPIOA, GPIO_PIN_4, MX28TYPE);
    Dynamixel_Init(&Motor13, 13, &huart3, GPIOB, GPIO_PIN_2, AX12ATYPE);
    Dynamixel_Init(&Motor14, 14, &huart3, GPIOB, GPIO_PIN_2, AX12ATYPE);
    Dynamixel_Init(&Motor15, 15, &huart3, GPIOB, GPIO_PIN_2, AX12ATYPE);
    Dynamixel_Init(&Motor16, 16, &huart3, GPIOB, GPIO_PIN_2, AX12ATYPE);
    Dynamixel_Init(&Motor17, 17, &huart3, GPIOB, GPIO_PIN_2, AX12ATYPE);
    Dynamixel_Init(&Motor18, 18, &huart3, GPIOB, GPIO_PIN_2, AX12ATYPE);


    Dynamixel_HandleTypeDef* arrDynamixel[18] = {&Motor1,&Motor2,&Motor3,&Motor4,
            &Motor5,&Motor6,&Motor7,&Motor8,&Motor9,&Motor10,&Motor11,&Motor12,
            &Motor13,&Motor14,&Motor15,&Motor16,&Motor17,&Motor18};

    UARTcmd_t Motorcmd[18];
    for(uint8_t i = MOTOR1; i <= MOTOR18; i++) {
        // Configure motor to return status packets only for read commands
        Dynamixel_SetStatusReturnLevel(arrDynamixel[i], 1);

        // Configure motor to return status packets with minimal latency
        Dynamixel_SetReturnDelayTime(arrDynamixel[i], 100);

        // Enable motor torque
        Dynamixel_TorqueEnable(arrDynamixel[i], 1);

        // Settings for torque near goal position, and acceptable error (AX12A only)
        if(arrDynamixel[i]->_motorType == AX12ATYPE){
            AX12A_SetComplianceSlope(arrDynamixel[i], 5); // 4 vibrates; 7 is too loose
            AX12A_SetComplianceMargin(arrDynamixel[i], 1);
        }

        (Motorcmd[i]).motorHandle = arrDynamixel[i];
        (Motorcmd[i]).type = cmdWritePosition;
    }

    (Motorcmd[MOTOR1]).qHandle = UART2_reqHandle;
    (Motorcmd[MOTOR2]).qHandle = UART2_reqHandle;
    (Motorcmd[MOTOR3]).qHandle = UART2_reqHandle;
    (Motorcmd[MOTOR4]).qHandle = UART4_reqHandle;
    (Motorcmd[MOTOR5]).qHandle = UART4_reqHandle;
    (Motorcmd[MOTOR6]).qHandle = UART4_reqHandle;
    (Motorcmd[MOTOR7]).qHandle = UART1_reqHandle;
    (Motorcmd[MOTOR8]).qHandle = UART1_reqHandle;
    (Motorcmd[MOTOR9]).qHandle = UART1_reqHandle;
    (Motorcmd[MOTOR10]).qHandle = UART6_reqHandle;
    (Motorcmd[MOTOR11]).qHandle = UART6_reqHandle;
    (Motorcmd[MOTOR12]).qHandle = UART6_reqHandle;
    (Motorcmd[MOTOR13]).qHandle = UART3_reqHandle;
    (Motorcmd[MOTOR14]).qHandle = UART3_reqHandle;
    (Motorcmd[MOTOR15]).qHandle = UART3_reqHandle;
    (Motorcmd[MOTOR16]).qHandle = UART3_reqHandle;
    (Motorcmd[MOTOR17]).qHandle = UART3_reqHandle;
    (Motorcmd[MOTOR18]).qHandle = UART3_reqHandle;

    Dynamixel_SetIOType(IO_DMA); // Configure IO to use DMA

    // Previous .c IMU initialization:
//    IMUdata._I2C_Handle = &hi2c1;
//    MPU6050_init(&IMUdata);
//    MPU6050_manually_set_offsets(&IMUdata);
//    MPU6050_set_LPF(&IMUdata, 4);

    IMUdata.init(4);

    // Set setupIsDone and unblock the higher-priority tasks
    setupIsDone = true;
    xTaskNotify(rxTaskHandle, 1UL, eNoAction);
    xTaskNotify(txTaskHandle, 1UL, eNoAction);
    xTaskNotify(UART1_Handle, 1UL, eNoAction);
    xTaskNotify(UART2_Handle, 1UL, eNoAction);
    xTaskNotify(UART3_Handle, 1UL, eNoAction);
    xTaskNotify(UART4_Handle, 1UL, eNoAction);
    xTaskNotify(UART6_Handle, 1UL, eNoAction);
    xTaskNotify(IMUTaskHandle, 1UL, eNoAction);

    /* Infinite loop */
    uint32_t numIterations = 0;
    uint8_t i;
    float positions[18];
    while(1){
        xTaskNotifyWait(0, NOTIFIED_FROM_TASK, NULL, portMAX_DELAY);

        // Convert raw bytes from robotGoal received from PC into floats
        for(uint8_t i = 0; i < 18; i++){
            uint8_t* ptr = (uint8_t*)&positions[i];
            for(uint8_t j = 0; j < 4; j++){
                *ptr = robotGoal.msg[i * 4 + j];
                ptr++;
            }
        }

        // TODO: This didn't really have an effect. Need to figure out why
        if(numIterations % 100 == 0){
            // Every 100 iterations, assert torque enable
            for(uint8_t i = MOTOR1; i <= MOTOR18; i++){
                Motorcmd[i].type = cmdWriteTorque;
                Motorcmd[i].value = 1; // Enable
                xQueueSend(Motorcmd[i].qHandle, &Motorcmd[i], 0);
            }
        }

        // Send each goal position to the queue, where the UART handler
        // thread that's listening will receive it and send it to the motor
        for(i = MOTOR1; i <= MOTOR18; i++){ // NB: i begins at 0 (i.e. Motor1 corresponds to i = 0)
            switch(i){
                case MOTOR1: Motorcmd[i].value = positions[i]*180/PI + 150;
                    break;
                case MOTOR2: Motorcmd[i].value = positions[i]*180/PI + 150;
                    break;
                case MOTOR3: Motorcmd[i].value = positions[i]*180/PI + 150;
                    break;
                case MOTOR4: Motorcmd[i].value = -1*positions[i]*180/PI + 150;
                    break;
                case MOTOR5: Motorcmd[i].value = -1*positions[i]*180/PI + 150;
                    break;
                case MOTOR6: Motorcmd[i].value = -1*positions[i]*180/PI + 150;
                    break;
                case MOTOR7: Motorcmd[i].value = -1*positions[i]*180/PI + 150;
                    break;
                case MOTOR8: Motorcmd[i].value = -1*positions[i]*180/PI + 150;
                    break;
                case MOTOR9: Motorcmd[i].value = positions[i]*180/PI + 150;
                    break;
                case MOTOR10: Motorcmd[i].value = -1*positions[i]*180/PI + 150;
                    break;
                case MOTOR11: Motorcmd[i].value = -1*positions[i]*180/PI + 150;
                    break;
                case MOTOR12: Motorcmd[i].value = positions[i]*180/PI + 150;
                    break;
                case MOTOR13: Motorcmd[i].value = positions[i]*180/PI + 150; // Left shoulder
                    break;
                case MOTOR14: Motorcmd[i].value = positions[i]*180/PI + 60; // Left elbow
                    break;
                case MOTOR15: Motorcmd[i].value = -1*positions[i]*180/PI + 150; // Right shoulder
                    break;
                case MOTOR16: Motorcmd[i].value = -1*positions[i]*180/PI + 240; // Right elbow
                    break;
                case MOTOR17: Motorcmd[i].value = -1*positions[i]*180/PI + 150; // Neck pan
                    break;
                case MOTOR18: Motorcmd[i].value = -1*positions[i]*180/PI + 150; // Neck tilt
                    break;
                default:
                    break;
            }

            Motorcmd[i].type = cmdWritePosition;
            xQueueSend(Motorcmd[i].qHandle, &Motorcmd[i], 0);

            // Only read from legs
            if(i <= MOTOR12){
                Motorcmd[i].type = cmdReadPosition;
                xQueueSend(Motorcmd[i].qHandle, &Motorcmd[i], 0);
            }
        }

        numIterations++;
    }
    /* USER CODE END StartDefaultTask */
}

/* UART1_Handler function */
/**
  * @brief  This function is executed in the context of the UART1_
  *         thread. It processes all commands for the motors
  *         physically connected to UART1, and initiates the I/O
  *         calls to them. Whenever it processes read commands for
  *         a motor, it sends the data received to the
  *         multi-writer sensor queue, which is read only by the
  *         TX task.
  *
  *         This function never returns.
  *
  * @ingroup Threads
  */
void UART1_Handler(void const * argument)
{
    /* USER CODE BEGIN UART1_Handler */
    // Here, we use task notifications to block this task from running until a notification
    // is received. This allows one-time setup to complete in a low-priority task.
    xTaskNotifyWait(UINT32_MAX, UINT32_MAX, NULL, portMAX_DELAY);

    /* Infinite loop */
    UARTcmd_t cmdMessage;
    TXData_t dataToSend;
    dataToSend.eDataType = eMotorData;

    for(;;)
    {
        while(xQueueReceive(UART1_reqHandle, &cmdMessage, portMAX_DELAY) != pdTRUE);
        UART_ProcessEvent(&cmdMessage, &dataToSend);
    }
    /* USER CODE END UART1_Handler */
}

/* UART2_Handler function */
/**
  * @brief  This function is executed in the context of the UART2_
  *         thread. It processes all commands for the motors
  *         physically connected to UART2, and initiates the I/O
  *         calls to them. Whenever it processes read commands for
  *         a motor, it sends the data received to the
  *         multi-writer sensor queue, which is read only by the
  *         TX task.
  *
  *         This function never returns.
  *
  * @ingroup Threads
  */
void UART2_Handler(void const * argument)
{
    /* USER CODE BEGIN UART2_Handler */
    // Here, we use task notifications to block this task from running until a notification
    // is received. This allows one-time setup to complete in a low-priority task.
    xTaskNotifyWait(UINT32_MAX, UINT32_MAX, NULL, portMAX_DELAY);

    /* Infinite loop */
    UARTcmd_t cmdMessage;
    TXData_t dataToSend;
    dataToSend.eDataType = eMotorData;

    for(;;)
    {
        while(xQueueReceive(UART2_reqHandle, &cmdMessage, portMAX_DELAY) != pdTRUE);
        UART_ProcessEvent(&cmdMessage, &dataToSend);
    }
    /* USER CODE END UART2_Handler */
}

/* UART3_Handler function */
/**
  * @brief  This function is executed in the context of the UART3_
  *         thread. It processes all commands for the motors
  *         physically connected to UART3, and initiates the I/O
  *         calls to them. Whenever it processes read commands for
  *         a motor, it sends the data received to the
  *         multi-writer sensor queue, which is read only by the
  *         TX task.
  *
  *         This function never returns.
  *
  * @ingroup Threads
  */
void UART3_Handler(void const * argument)
{
    /* USER CODE BEGIN UART3_Handler */
    // Here, we use task notifications to block this task from running until a notification
    // is received. This allows one-time setup to complete in a low-priority task.
    xTaskNotifyWait(UINT32_MAX, UINT32_MAX, NULL, portMAX_DELAY);

    /* Infinite loop */
    UARTcmd_t cmdMessage;
    TXData_t dataToSend;
    dataToSend.eDataType = eMotorData;

    for(;;)
    {
        while(xQueueReceive(UART3_reqHandle, &cmdMessage, portMAX_DELAY) != pdTRUE);
        UART_ProcessEvent(&cmdMessage, &dataToSend);
    }
    /* USER CODE END UART3_Handler */
}

/* UART4_Handler function */
/**
  * @brief  This function is executed in the context of the UART4_
  *         thread. It processes all commands for the motors
  *         physically connected to UART4, and initiates the I/O
  *         calls to them. Whenever it processes read commands for
  *         a motor, it sends the data received to the
  *         multi-writer sensor queue, which is read only by the
  *         TX task.
  *
  *         This function never returns.
  *
  * @ingroup Threads
  */
void UART4_Handler(void const * argument)
{
    /* USER CODE BEGIN UART4_Handler */
    // Here, we use task notifications to block this task from running until a notification
    // is received. This allows one-time setup to complete in a low-priority task.
    xTaskNotifyWait(UINT32_MAX, UINT32_MAX, NULL, portMAX_DELAY);

    /* Infinite loop */
    UARTcmd_t cmdMessage;
    TXData_t dataToSend;
    dataToSend.eDataType = eMotorData;

    for(;;)
    {
        while(xQueueReceive(UART4_reqHandle, &cmdMessage, portMAX_DELAY) != pdTRUE);
        UART_ProcessEvent(&cmdMessage, &dataToSend);
    }
    /* USER CODE END UART4_Handler */
}

/* UART6_Handler function */
/**
  * @brief  This function is executed in the context of the UART6_
  *         thread. It processes all commands for the motors
  *         physically connected to UART6, and initiates the I/O
  *         calls to them. Whenever it processes read commands for
  *         a motor, it sends the data received to the
  *         multi-writer sensor queue, which is read only by the
  *         TX task.
  *
  *         This function never returns.
  *
  * @ingroup Threads
  */
void UART6_Handler(void const * argument)
{
    /* USER CODE BEGIN UART6_Handler */
    // Here, we use task notifications to block this task from running until a notification
    // is received. This allows one-time setup to complete in a low-priority task.
    xTaskNotifyWait(UINT32_MAX, UINT32_MAX, NULL, portMAX_DELAY);

    /* Infinite loop */
    UARTcmd_t cmdMessage;
    TXData_t dataToSend;
    dataToSend.eDataType = eMotorData;

    for(;;)
    {
        while(xQueueReceive(UART6_reqHandle, &cmdMessage, portMAX_DELAY) != pdTRUE);
        UART_ProcessEvent(&cmdMessage, &dataToSend);
    }
    /* USER CODE END UART6_Handler */
}

/* StartIMUTask function */
/**
  * @brief  This function is executed in the context of the
  *         IMUTask thread. During each control cycle, this thread
  *         fetches accelerometer and gyroscope data, then sends
  *         this data to the multi-writer sensor queue, which is
  *         read only by the TX task.
  *
  *         This function never returns.
  *
  * @ingroup Threads
  */
void StartIMUTask(void const * argument)
{
  /* USER CODE BEGIN StartIMUTask */
  // Here, we use task notifications to block this task from running until a notification
  // is received. This allows one-time setup to complete in a low-priority task.
  xTaskNotifyWait(UINT32_MAX, UINT32_MAX, NULL, portMAX_DELAY);

  TXData_t dataToSend;
  dataToSend.eDataType = eIMUData;

  TickType_t xLastWakeTime;
  xLastWakeTime = xTaskGetTickCount();

  const TickType_t IMU_CYCLE_TIME_MS = 2;
  uint8_t i = 0;
  IMUStruct IMUStruct;

  MPUFilter_InitAllFilters();

  for(;;)
  {
      // Service this thread every 2 ms for a 500 Hz sample rate
      vTaskDelayUntil(&xLastWakeTime, pdMS_TO_TICKS(IMU_CYCLE_TIME_MS));

      IMUdata.Read_Accelerometer_Withoffset_IT(); // Also updates pitch and roll

      // Gyroscope data is much more volatile/sensitive to changes than
      // acceleration data. To compensate, we feed in samples to the filter
      // slower. Good DSP practise? Not sure. To compensate for the high
      // delays, we also use a filter with fewer taps than the acceleration
      // filters. Ideally: we would sample faster to reduce aliasing, then
      // use a filter with a smaller cutoff frequency. However, the filter
      // designer we are using does not allow us to generate such filters in
      // the free version, so this is the best we can do unless we use other
      // software.
      if (i % 16 == 0) {
    	  IMUdata.Read_Gyroscope_Withoffset_IT();
// TODO: convert the MPUFilter_FilterAngularVelocity function
          //MPUFilter_FilterAngularVelocity();
      }
      i++;
      IMUdata.Fill_Struct(&IMUStruct);
      dataToSend.pData = &IMUStruct;
      xQueueSend(TXQueueHandle, &dataToSend, 0);
  }
  /* USER CODE END StartIMUTask */
}

/* StartRxTask function */
/**
  * @brief  This function is executed in the context of the RxTask
  *         thread. It initiates DMA-based receptions of RobotGoals
  *         from the PC via UART5. Upon successful reception of a
  *         RobotGoal, the UARTx_ and IMUTask threads are unblocked.
  *
  *         This function never returns.
  *
  * @ingroup Threads
  */
void StartRxTask(void const * argument)
{
    /* USER CODE BEGIN StartRxTask */
    uint8_t robotGoalData[sizeof(RobotGoal)];
    uint8_t *robotGoalDataPtr;
    uint8_t buffRx[92];
    uint8_t startSeqCount;
    uint8_t totalBytesRead;

    // Receiving
    robotGoal.id = 0;
    robotGoalDataPtr = robotGoalData;
    startSeqCount = 0;
    totalBytesRead = 0;

    // Sending
    robotState.id = 0;
    robotState.start_seq = UINT32_MAX;
    robotState.end_seq = 0;

    xTaskNotifyWait(UINT32_MAX, UINT32_MAX, NULL, portMAX_DELAY);

    HAL_StatusTypeDef status;

    uint32_t notification;

    HAL_UART_Receive_DMA(&huart5, (uint8_t*)buffRx, sizeof(buffRx));

    /* Infinite loop */
    for (;;)
    {
        // Wait until notified from ISR. Clear no bits on entry in case the notification
        // comes before this statement is executed (which is rather unlikely as long as
        // this task has the highest priority, but overall this is a better decision in
        // case priorities are changed in the future and someone forgets about this.
        do{
            xTaskNotifyWait(0, NOTIFIED_FROM_RX_ISR, &notification, portMAX_DELAY);
        }while((notification & NOTIFIED_FROM_RX_ISR) != NOTIFIED_FROM_RX_ISR);

        do{
            // This do-while loop with the mutex inside of it makes calls to the UART module
            // responsible for PC communication atomic. This attempts to solve the following
            // scenario: the TX thread is in the middle of executing the call to HAL_UART_Transmit
            // when suddenly the RX thread is unblocked. The RX thread calls HAL_UART_Receive, and
            // returns immediately when it detects that the uart module is already locked. Then
            // the RX thread blocks itself and never wakes up since a RX transfer was never
            // initialized.
            xSemaphoreTake(PCUARTHandle, 1);
            status = HAL_UART_Receive_DMA(&huart5, (uint8_t*)buffRx, sizeof(buffRx));
            xSemaphoreGive(PCUARTHandle);
        }while(status != HAL_OK);

        for (uint8_t i = 0; i < sizeof(buffRx); i++) {
            if (startSeqCount == 4) {
                // This control block is entered when the header sequence of
                // 0xFFFFFFFF has been received; thus we know the data we
                // receive will be in tact

                *robotGoalDataPtr = buffRx[i];
                robotGoalDataPtr++;
                totalBytesRead++;

                if (totalBytesRead == sizeof(RobotGoal)) {
                    // If, after the last couple of receive interrupts, we have
                    // received sizeof(RobotGoal) bytes, then we copy the data
                    // buffer into the robotGoal structure and wake the control
                    // thread to distribute states to each actuator
                    memcpy(&robotGoal, robotGoalData, sizeof(RobotGoal));
                    robotState.id = robotGoal.id;

                    // Reset the variables to help with reception of a RobotGoal
                    robotGoalDataPtr = robotGoalData;
                    startSeqCount = 0;
                    totalBytesRead = 0;

                    xTaskNotify(defaultTaskHandle, NOTIFIED_FROM_TASK, eSetBits); // Wake control task
                    xTaskNotify(IMUTaskHandle, NOTIFIED_FROM_TASK, eSetBits); // Wake MPU task
                    continue;
                }
            }else{
                // This control block is used to verify that the data header is in tact
                if (buffRx[i] == 0xFF) {
                    startSeqCount++;
                } else {
                    startSeqCount = 0;
                }
            }
        }
    }
    /* USER CODE END StartRxTask */
}

/* StartTxTask function */
/**
  * @brief  This function is executed in the context of the TxTask
  *         thread. This thread is blocked until all sensor data
  *         has been received through the sensor queue. After this
  *         time, the UARTx_ and IMUTask will be blocked. Then, a
  *         DMA-based transmission of a RobotState is sent to the
  *         PC via UART5.
  *
  *         This function never returns.
  *
  * @ingroup Threads
  */
void StartTxTask(void const * argument)
{
    /* USER CODE BEGIN StartTxTask */
    xTaskNotifyWait(UINT32_MAX, UINT32_MAX, NULL, portMAX_DELAY);

    TXData_t receivedData;
    Dynamixel_HandleTypeDef* motorPtr = NULL;
    IMUStruct* imuPtr = NULL;
    char* const pIMUXGyroData = &robotState.msg[ROBOT_STATE_MPU_DATA_OFFSET];

    HAL_StatusTypeDef status;
    uint32_t notification;
    uint32_t dataReadyFlags = 0; // Bits in this are set based on which sensor data is ready

    // TODO: In the future, this "12" should be replaced with NUM_MOTORS. We will
    // be ready for this once all 18 motors can be read from.
    uint32_t NOTIFICATION_MASK = 0x80000000;
    for(uint8_t i = 1; i <= 12; i++){
        NOTIFICATION_MASK |= (1 << i);
    }

    /* Infinite loop */
    for(;;)
    {
        while((dataReadyFlags & NOTIFICATION_MASK) != NOTIFICATION_MASK){
            while(xQueueReceive(UART_rxHandle, &receivedData, portMAX_DELAY) != pdTRUE);

            switch(receivedData.eDataType){
            case eMotorData:
                motorPtr = (Dynamixel_HandleTypeDef*)receivedData.pData;

                if(motorPtr == NULL){ break; }

                // Validate data and store it in robotState
                if(motorPtr->_ID <= NUM_MOTORS){
                    // Copy sensor data for this motor into its section of robotState.msg
                    memcpy(&robotState.msg[4 * (motorPtr->_ID - 1)], &(motorPtr->_lastPosition), sizeof(float));

                    // Set flag indicating the motor with this id has reported in with position data
                    dataReadyFlags |= (1 << motorPtr->_ID);
                }
                break;
            case eIMUData:
                imuPtr = (IMUStruct*)receivedData.pData;

                if(imuPtr == NULL){ break; }

                // Copy sensor data into the IMU data section of robotState.msg
                memcpy(pIMUXGyroData, (&imuPtr->_x_Gyro), 6 * sizeof(float));

                // Set flag indicating IMU data has reported in
                dataReadyFlags |= 0x80000000;
                break;
            default:
                break;
            }
        }
        dataReadyFlags = 0; // Clear all flags

        do{
            // This do-while loop with the mutex inside of it makes calls to the UART module
            // responsible for PC communication atomic. This attempts to solve the following
            // scenario: the TX thread is in the middle of executing the call to HAL_UART_Transmit
            // when suddenly the RX thread is unblocked. The RX thread calls HAL_UART_Receive, and
            // returns immediately when it detects that the uart module is already locked. Then
            // the RX thread blocks itself and never wakes up since a RX transfer was never
            // initialized.
            xSemaphoreTake(PCUARTHandle, 1);
            status = HAL_UART_Transmit_DMA(&huart5, (uint8_t*)&robotState, sizeof(RobotState));
            xSemaphoreGive(PCUARTHandle);
        }while(status != HAL_OK);

        // Wait until notified from ISR. Clear no bits on entry in case the notification
        // came while a higher priority task was executing.
        do{
            xTaskNotifyWait(0, NOTIFIED_FROM_TX_ISR, &notification, portMAX_DELAY);
        }while((notification & NOTIFIED_FROM_TX_ISR) != NOTIFIED_FROM_TX_ISR);
    }
    /* USER CODE END StartTxTask */
}

/* USER CODE BEGIN Application */
/**
 * @defgroup Callbacks Callbacks
 * @brief    Callback functions for unblocking FreeRTOS threads which perform
 *           non-blocking I/O
 *
 * @ingroup FreeRTOS
 */

/**
  * @brief  This function is called whenever a memory read from a I2C
  *         device is completed. For this program, the callback behaviour
  *         consists of unblocking the thread which initiated the I/O and
  *         yielding to a higher priority task from the ISR if there are
  *         any that can run.
  * @param  hi2c pointer to a I2C_HandleTypeDef structure that contains
  *         the configuration information for I2C module corresponding to
  *         the callback
  * @return None
  *
  * @ingroup Callbacks
  */
void HAL_I2C_MemRxCpltCallback(I2C_HandleTypeDef *hi2c)
{
	// This callback runs after the interrupt data transfer from the sensor to the mcu is finished
	BaseType_t xHigherPriorityTaskWoken = pdFALSE;
	xTaskNotifyFromISR(IMUTaskHandle, NOTIFIED_FROM_RX_ISR, eSetBits, &xHigherPriorityTaskWoken);
	portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

/**
  * @brief  This function is called whenever a transmission from a UART
  *         module is completed. For this program, the callback behaviour
  *         consists of unblocking the thread which initiated the I/O and
  *         yielding to a higher priority task from the ISR if there are
  *         any that can run.
  * @param  huart pointer to a UART_HandleTypeDef structure that contains
  *         the configuration information for UART module corresponding to
  *         the callback
  * @return None
  *
  * @ingroup Callbacks
  */
void HAL_UART_TxCpltCallback(UART_HandleTypeDef* huart){
    if(setupIsDone){
        BaseType_t xHigherPriorityTaskWoken = pdFALSE;
        if(huart == &huart5){
            xTaskNotifyFromISR(txTaskHandle, NOTIFIED_FROM_TX_ISR, eSetBits, &xHigherPriorityTaskWoken);
        }
        if(huart == &huart1){
            xTaskNotifyFromISR(UART1_Handle, NOTIFIED_FROM_TX_ISR, eSetBits, &xHigherPriorityTaskWoken);
        }
        else if(huart == &huart2){
            xTaskNotifyFromISR(UART2_Handle, NOTIFIED_FROM_TX_ISR, eSetBits, &xHigherPriorityTaskWoken);
        }
        else if(huart == &huart3){
            xTaskNotifyFromISR(UART3_Handle, NOTIFIED_FROM_TX_ISR, eSetBits, &xHigherPriorityTaskWoken);
        }
        else if(huart == &huart4){
            xTaskNotifyFromISR(UART4_Handle, NOTIFIED_FROM_TX_ISR, eSetBits, &xHigherPriorityTaskWoken);
        }
        else if(huart == &huart6){
            xTaskNotifyFromISR(UART6_Handle, NOTIFIED_FROM_TX_ISR, eSetBits, &xHigherPriorityTaskWoken);
        }
        portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
    }
}

/**
  * @brief  This function is called whenever a reception from a UART
  *         module is completed. For this program, the callback behaviour
  *         consists of unblocking the thread which initiated the I/O and
  *         yielding to a higher priority task from the ISR if there are
  *         any that can run.
  * @param  huart pointer to a UART_HandleTypeDef structure that contains
  *         the configuration information for UART module corresponding to
  *         the callback
  * @return None
  *
  * @ingroup Callbacks
  */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef * huart) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    if (huart == &huart5) {
        xTaskNotifyFromISR(rxTaskHandle, NOTIFIED_FROM_RX_ISR, eSetBits, &xHigherPriorityTaskWoken);
    }
    if(huart == &huart1){
        xTaskNotifyFromISR(UART1_Handle, NOTIFIED_FROM_RX_ISR, eSetBits, &xHigherPriorityTaskWoken);
    }
    else if(huart == &huart2){
        xTaskNotifyFromISR(UART2_Handle, NOTIFIED_FROM_RX_ISR, eSetBits, &xHigherPriorityTaskWoken);
    }
    else if(huart == &huart3){
        xTaskNotifyFromISR(UART3_Handle, NOTIFIED_FROM_RX_ISR, eSetBits, &xHigherPriorityTaskWoken);
    }
    else if(huart == &huart4){
        xTaskNotifyFromISR(UART4_Handle, NOTIFIED_FROM_RX_ISR, eSetBits, &xHigherPriorityTaskWoken);
    }
    else if(huart == &huart6){
        xTaskNotifyFromISR(UART6_Handle, NOTIFIED_FROM_RX_ISR, eSetBits, &xHigherPriorityTaskWoken);
    }
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

/**
  * @brief  This function is called whenever an error is encountered in
  *         association with a UART module. For this program, the callback
  *         behaviour consists of storing the error code in a local
  *         variable.
  * @param  huart pointer to a UART_HandleTypeDef structure that contains
  *         the configuration information for UART module corresponding to
  *         the callback
  * @return None
  *
  * @ingroup Callbacks
  */
void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
    error = HAL_UART_GetError(huart);
}
/* USER CODE END Application */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
