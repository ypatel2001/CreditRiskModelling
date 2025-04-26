package handlers

import (
	"bytes"
	"net/http"
	"os/exec"

	"github.com/gin-gonic/gin"
)

func Hello(c *gin.Context) {
	// Call the Python hello.py script
	cmd := exec.Command("/opt/miniconda3/bin/python3", "../model/hello.py")
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to run Python script"})
		return
	}

	// Return the script's output
	c.JSON(http.StatusOK, gin.H{"message": out.String()})
}
