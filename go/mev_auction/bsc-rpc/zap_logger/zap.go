package zap_logger

import (
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"gopkg.in/natefinch/lumberjack.v2"
	"os"
)

var Zap *zap.Logger

func init() {
	// 配置 lumberjack 日志轮转
	lumberjackLogger := &lumberjack.Logger{
		Filename:   "./log/rpc.log", // 日志文件路径
		MaxSize:    200,             // 每个日志文件的大小，单位是MB
		MaxBackups: 200,             // 最多保留200个备份日志文件
		MaxAge:     30,              // 保留日志的天数
		Compress:   false,           // 压缩旧日志文件
	}

	// 创建编码器（格式）
	encoderConfig := zap.NewProductionEncoderConfig()
	encoderConfig.TimeKey = "timestamp"
	encoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder // 使用 ISO8601 时间格式

	consoleEncoder := zapcore.NewConsoleEncoder(encoderConfig)
	fileEncoder := zapcore.NewJSONEncoder(encoderConfig)

	// 创建多个输出目标
	core := zapcore.NewTee(
		zapcore.NewCore(consoleEncoder, zapcore.AddSync(os.Stdout), zapcore.InfoLevel),     // 控制台输出
		zapcore.NewCore(fileEncoder, zapcore.AddSync(lumberjackLogger), zapcore.InfoLevel), // 文件输出，使用 lumberjack 进行文件轮转
	)

	// 创建 logger
	Zap = zap.New(core)
}
