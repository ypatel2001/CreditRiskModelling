package main

import (
	"credit-risk-api/handlers"

	"github.com/gin-gonic/gin"
)

func main() {
	r := gin.Default()
	r.GET("/hello", handlers.Hello)
	r.POST("/predict", handlers.Predict)
	r.Run(":8080")
}
